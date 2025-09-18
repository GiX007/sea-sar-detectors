# src/detr.py
#
# Original DETR paper: https://arxiv.org/abs/2005.12872
#
# Using Hugging Face DETR, wrapped to behave like a TorchVision detector:
# - train mode: model(images, targets) -> dict of losses
# - eval  mode: model(images) -> list of dicts with "boxes","labels","scores"
#
import os, time, torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.utils import plot_training_curves, count_parameters, train_one_epoch, validate_one_epoch, iou_metrics

import warnings
from transformers.utils import logging as hf_logging
warnings.filterwarnings("ignore", message=".*meta parameter.*no-op.*")
hf_logging.set_verbosity_error()


class _DetrTorchvisionAdapter(nn.Module):
    """
    Wraps HF DETR so it matches TorchVision's detection API used elsewhere.

    Label conventions bridged:
      - Dataset emits labels in [1..C-1] with 0=background (total classes = C)
      - HF DETR wants labels in [0..C-2] (no explicit background)
      - We shift targets (1..C-1) -> (0..C-2) for loss, and predictions back (0..C-2) -> (1..C-1)

    Boxes:
      - We feed absolute xyxy pixels (your dataset already resizes images and scales boxes)
      - HF post-process also returns absolute xyxy; we pass them through

    Train/Eval contract:
      - Train/Val: call in train() and pass targets -> returns dict of losses
      - Eval: call in eval() without targets -> returns list of {boxes, scores, labels}
    """
    def __init__(self, num_classes: int, device: torch.device):
        """
        Args:
            num_classes: total classes INCLUDING the background (C). Foreground count = C-1
            device: torch device to run on
        """
        super().__init__()
        assert num_classes >= 2, "Need background + at least one foreground class."
        num_foreground = num_classes - 1  # HF DETR expects this number

        # Load COCO-pretrained DETR; resize classification head to foreground count
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_foreground,  # foreground classes only (= num_classes - 1)
            ignore_mismatched_sizes=True,
        )
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        # self.processor.size = {"shortest_edge": 256}
        self.processor.do_resize = False
        self.processor.do_rescale = False
        self.processor.do_normalize = False

        # freeze the CNN backbone for faster backprop on small GPU
        try:
            for p in self.model.model.backbone.parameters():
                p.requires_grad = False
        except Exception:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        self.device = device
        self.model.to(device)

    def _targets_to_hf(self, targets):
        """
        Convert TorchVision targets -> COCO detection for DetrImageProcessor.

        Input (per image):
          boxes: Float[N,4] absolute xyxy
          labels: Long[N] in [1..C-1]  (0 is background, not used)
          (optional) image_id: int or 0-dim tensor

        Output (per image):
          {
            "image_id": int,
            "annotations": [
              {"bbox": [x, y, w, h], "category_id": int(label_0_based)}, ...
            ]
          }
        """
        coco_list = []
        for i, t in enumerate(targets):
            boxes_xyxy = t["boxes"]
            labels_1 = t["labels"]

            labels_0 = labels_1 - 1 if labels_1.numel() > 0 else labels_1

            if boxes_xyxy.numel() > 0:
                x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
                w = (x2 - x1).clamp(min=0)
                h = (y2 - y1).clamp(min=0)
                boxes_xywh = torch.stack([x1, y1, w, h], dim=1)

                anns = []
                for b, c in zip(boxes_xywh, labels_0):
                    x, y, ww, hh = [float(v) for v in b.tolist()]
                    anns.append({
                        "bbox": [x, y, ww, hh],
                        "category_id": int(c),
                        "iscrowd": 0,
                        "area": float(ww * hh),
                    })
            else:
                anns = []

            img_id = t.get("image_id", i)
            if isinstance(img_id, torch.Tensor):
                img_id = int(img_id.item())
            else:
                img_id = int(img_id)

            coco_list.append({"image_id": img_id, "annotations": anns})

        return coco_list

    @torch.no_grad()
    def _postprocess(self, outputs, images):
        """
        Converts HF outputs to TorchVision-like dicts with labels shifted back to [1..C-1].

        Args:
            outputs: DETR raw outputs
            images: list[Tensor 3xHxW] used only to get target sizes

        Returns:
            list[dict]: {"boxes": Float[M,4], "scores": Float[M], "labels": Long[M] in [1..C-1]}
        """
        # target_sizes expects (h, w) per image
        sizes = [img.shape[-2:] for img in images]
        target_sizes = [(int(h), int(w)) for (h, w) in sizes]

        # Threshold filters low-confidence boxes before NMS inside processor
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )

        tv_results = []
        for r in results:
            tv_results.append({
                "boxes":  r["boxes"].to(self.device), # [N,4] xyxy abs
                "scores": r["scores"].to(self.device), # [N]
                "labels": (r["labels"] + 1).to(self.device).long(), # [N], shift back to 1..C-1
            })
        return tv_results

    def forward(self, images, targets=None):
        """
        Train:
          images: list[Tensor 3xHxW], [0,1]
          targets: list[{"boxes": Float[N,4] xyxy abs, "labels": Long[N] 1..C-1}]
          returns: dict of losses (e.g., loss_ce, loss_bbox, loss_giou)

        Eval:
          images: list[Tensor]
          returns: list[{"boxes","scores","labels"}] with labels in 1..C-1
        """
        # If targets are provided, do the loss path (train/val loss)
        if targets is not None:
            hf_targets = self._targets_to_hf(targets)
            enc = self.processor(images=images, annotations=hf_targets, return_tensors="pt")

            pixel_values = enc["pixel_values"].to(self.device)
            pixel_mask = enc.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(self.device)

            labels = enc["labels"]  # list[dict]
            # >>> move any tensor fields inside each label dict to the model device
            for t in labels:
                for k, v in list(t.items()):
                    if torch.is_tensor(v):
                        t[k] = v.to(self.device)

            outputs = self.model(pixel_values=pixel_values,
                                 pixel_mask=pixel_mask,
                                 labels=labels)

            loss_total = outputs.loss
            loss_dict = getattr(outputs, "loss_dict", None) or {"loss": loss_total}
            return loss_dict

        # Otherwise, do the prediction path (eval inference)
        with torch.no_grad():
            enc = self.processor(images=images, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(self.device)
            pixel_mask = enc.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(self.device)

            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            return self._postprocess(outputs, images)


def build_detr(num_classes, device: torch.device) -> nn.Module:
    """
    Builds DETR wrapped with TorchVision-like API.

    Args:
        num_classes (int): total classes INCLUDING background (C = 1 + K)
        device: torch device

    Returns:
        nn.Module
    """
    return _DetrTorchvisionAdapter(num_classes=num_classes, device=device)


def run_detr(
    train_loader, val_loader, test_loader, num_classes, device,
    epochs = 200, lr = 1e-4, weight_decay = 1e-4, results_dir = "results",
    early_stopping_patience = 10, early_stopping_min_delta = 1e-4,
):
    """
    Trains, validates (early-stop on val IoU@0.5), saves curves, and tests DETR.

    Args:
        train_loader: train detection dataloader (lists of images/targets)
        val_loader: validation detection dataloader (lists of images/targets)
        test_loader: test detection dataloader (lists of images/targets)
        num_classes: total classes INCLUDING background (C)
        device: torch.device
        epochs: training epochs
        lr: learning rate
        weight_decay: L2 Regularization
        results_dir: where to write logs/plots
        early_stopping_patience: stop if val_iou stalls
        early_stopping_min_delta: min improvement to reset patience

    Returns:
        list[dict]: single-row metrics summary
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_processes.txt")
    fig_dir = os.path.join(results_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    curves_path = os.path.join(fig_dir, "detr_training_curves.png")

    # Model + optimizer
    model = build_detr(num_classes, device=device).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    # Early stopping state
    best_val_iou = -float("inf")
    best_state_dict = None
    patience_counter = 0
    best_epoch = 0

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n>>> Training of DETR ...\n")
        f.write(f"Epochs: {epochs} | LR: {lr} | Optimizer: AdamW | Weight Decay: {weight_decay}\n")
        f.write(f"EarlyStopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}\n\n")

    # Train and Validate
    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device)

        # Validate (loss + val IoU)
        val_loss, val_iou = validate_one_epoch(model, val_loader, device)

        # Train IoU — eval pass on train set for monitoring
        # model.eval()
        # iou_sum, n_imgs = 0.0, 0
        # with torch.no_grad():
        #     for images, targets in train_loader:
        #         images  = [img.to(device) for img in images]
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #         outputs = model(images)
        #         for out, tgt in zip(outputs, targets):
        #             pred_boxes = out.get("boxes", torch.empty(0, 4, device=device))
        #             gt_boxes   = tgt.get("boxes", torch.empty(0, 4, device=device))
        #             mean_iou, _, _ = iou_metrics(pred_boxes, gt_boxes, iou_thr=0.5)
        #             iou_sum += float(mean_iou)
        #             n_imgs  += 1
        # train_iou = iou_sum / max(1, n_imgs)
        train_iou = float("nan")

        # Record & log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch: {epoch} | "
                f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f} | "
                f"Train IoU: {train_iou:.6f}  Val IoU: {val_iou:.6f}\n"
            )

        # Early stopping on Val IoU
        improved = (val_iou - best_val_iou) > early_stopping_min_delta
        if improved:
            best_val_iou = val_iou
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best val_iou at epoch {best_epoch}: {best_val_iou:.6f})\n"
                    )
                break
    training_time_sec = time.time() - t0_train

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Params
    n_params = count_parameters(model)

    # Curves
    plot_training_curves(history, save_path=curves_path)

    # TEST: mAP@0.5 + IoU/Prec/Rec + avg inference time
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()

    total_iou = total_prec = total_rec = 0.0
    n_imgs = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images) # predictions already have labels in 1..C-1
            map_metric.update(outputs, targets)

            for out, tgt in zip(outputs, targets):
                pred_boxes = out.get("boxes", torch.empty(0, 4, device=device))
                gt_boxes = tgt.get("boxes", torch.empty(0, 4, device=device))
                mean_iou, precision, recall = iou_metrics(pred_boxes, gt_boxes, iou_thr=0.5)
                total_iou += float(mean_iou)
                total_prec += float(precision)
                total_rec += float(recall)
                n_imgs += 1

    iou05 = total_iou / max(1, n_imgs)
    precision05 = total_prec / max(1, n_imgs)
    recall05 = total_rec / max(1, n_imgs)
    result = map_metric.compute()
    map05 = result["map_50"].item()

    # AVG INFERENCE TIME
    total_time, total_imgs = 0.0, 0
    with torch.no_grad():
        for images, *_ in test_loader:
            images = [img.to(device) for img in images]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += (time.time() - t0)
            total_imgs += len(images)
    avg_inf_time = total_time / max(1, total_imgs)

    # Log training info
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"TEST IoU@0.5={iou05:.4f} Prec@0.5={precision05:.4f} Rec@0.5={recall05:.4f}\n")
        f.write(
            f"Training finished in {training_time_sec:.2f} seconds | "
            f"Trainable parameters: {n_params:,}\n"
        )

    # Save trained model
    torch.save(model.state_dict(), os.path.join(models_dir, "detr.pth"))

    row = {
        "Model": "DETR ", # (HF wrapped)
        "IoU@0.5": iou05,
        "Precision@0.5": precision05,
        "Recall@0.5": recall05,
        "mAP@0.5": map05,
        "Training Time (s)": training_time_sec,
        "Average Inference Time (s)": avg_inf_time,
        "#Params": n_params,
    }

    return [row]
