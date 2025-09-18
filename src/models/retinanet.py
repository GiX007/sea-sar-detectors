# src/models/retinanet.py
#
# Original RetinaNet paper: https://arxiv.org/abs/1708.02002
#
# Minimal RetinaNet runner built on TorchVision's detection API:
# https://pytorch.org/vision/stable/models/retinanet.html
#
# Key points:
# - Our dataset produces labels in [1..K] (0 is background, per TorchVision convention).
# - RetinaNet's head is sized for K foreground classes and its loss expects labels in [0..K-1].
#
import os, time, torch
import torchvision
from torchvision.models import ResNet50_Weights
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.utils import plot_training_curves, count_parameters, iou_metrics, train_one_epoch, validate_one_epoch


# RetinaNet adapter (1-based -> 0-based for loss; 0-based -> 1-based for preds)
class _RetinaTorchvisionAdapter(nn.Module):
    """
    Wraps torchvision RetinaNet so it plays nicely with this codebase's label space.

    Label conventions bridged:
      - Dataset/metrics expect labels in [1..K], with 0 reserved for the background
      - RetinaNet (and its loss) expect labels in [0..K-1] (foreground only)

    Behavior:
      - Train/val (when `targets` provided): shift labels from 1..K -> 0..K-1
      - Eval (no `targets`): shift predicted labels from 0..K-1 -> 1..K so the rest
        of the pipeline (mAP, IoU helper, etc.) remains consistent across models
    """
    def __init__(self, num_classes_including_bg: int):
        """
        Args:
            num_classes_including_bg: C_total = K + 1 (background at index 0)
        """
        super().__init__()
        assert num_classes_including_bg >= 2, "Need at least background + 1 class."
        k_fg = num_classes_including_bg - 1  # K foreground classes (what RetinaNet needs)

        # Build RetinaNet with IMAGENET-pretrained backbone and fresh classification head (no COCO 91-head)
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None,  # avoid the 91-class COCO head constraint
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            num_classes=k_fg,
        )

        # Freeze the ResNet backbone for speed
        for p in self.model.backbone.body.parameters():
            p.requires_grad = False

        # Light inference-time tweaks
        self.model.score_thresh = 0.2 # default 0.05; bump for fewer low-score boxes
        self.model.nms_thresh = 0.5 # keep default
        self.model.detections_per_img = 100 # default 300 -> fewer post-NMS boxes

        self.k_fg = k_fg # stored for basic sanity filtering

    @staticmethod
    def _shift_targets_1to0(targets, k_fg: int):
        """
        Shifts the target labels from [1..K] to [0..K-1] and drop any label outside that range.

        Args:
            targets (list[dict]): each has 'boxes' (Float[N,4]) and 'labels' (Long[N]) in [1..K]
            k_fg (int): number of foreground classes (K)

        Returns:
            list[dict]: labels in [0..K-1], boxes preserved (same tensors, same device)
        """
        out = []
        for t in targets:
            boxes = t["boxes"]
            labels = t["labels"]
            if labels.numel():
                labels0 = labels - 1  # 1..K -> 0..K-1
                mask = (labels0 >= 0) & (labels0 < k_fg)
                boxes = boxes[mask]
                labels0 = labels0[mask].long()
            else:
                labels0 = labels
            out.append({"boxes": boxes, "labels": labels0})
        return out

    @staticmethod
    def _shift_outputs_0to1(outputs):
        """
        Shifts predicted labels from [0..K-1] to [1..K].

        Args:
            outputs (list[dict]): each has 'boxes','scores','labels' (Long in 0..K-1)

        Returns:
            list[dict]: labels in 1..K (Long), boxes/scores unchanged
        """
        shifted = []
        for o in outputs:
            # Keep incoming tensors/devices; only adjust labels
            shifted.append({
                "boxes":  o["boxes"],
                "scores": o.get("scores", torch.zeros(0, device=o["boxes"].device)),
                "labels": (o["labels"].long() + 1),
            })
        return shifted

    def forward(self, images, targets=None):
        """
        TorchVision-like interface:
          - Train: model(images, targets) -> dict of losses
          - Eval: model(images) -> list[dict] with "boxes","labels","scores"

        We intercept to shift label spaces so the rest of your code can stay unchanged.
        """
        if targets is not None:
            # TRAIN / VAL (loss path): shift labels to 0-based
            targets_0 = self._shift_targets_1to0(targets, self.k_fg)
            return self.model(images, targets_0)

        # EVAL (prediction path): shift labels to 1-based so metrics align with other models
        with torch.no_grad():
            outputs = self.model(images)
            return self._shift_outputs_0to1(outputs)


# Build + Run

def build_retinanet(num_classes: int) -> nn.Module:
    """
    Builds RetinaNet (ResNet50-FPN) wrapped with label-space adapter.

    Args:
        num_classes: total classes INCLUDING the background (0 = background). For your data, this should be K + 1 where K excludes "ignored"

    Returns:
        nn.Module: wrapped RetinaNet ready to train/eval with your loaders
    """
    return _RetinaTorchvisionAdapter(num_classes)


def run_retinanet(
    train_loader, val_loader, test_loader, num_classes, device,
    epochs=100, lr=0.005, momentum=0.9, weight_decay=0.0005, results_dir="results",
    early_stopping_patience=10, early_stopping_min_delta=1e-4
):
    """
    Trains, validates, plots curves, and evaluates RetinaNet on the TEST set with early stopping on validation IoU@0.5.

    Args:
        train_loader: training dataloader yielding (images, targets)
        val_loader: validation dataloader yielding (images, targets)
        test_loader: test dataloader yielding (images, targets)
        num_classes: number of classes INCLUDING background
        device: torch device ('cuda' or 'cpu')
        epochs: number of training epochs
        lr: learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization
        results_dir: directory to save logs/plots
        early_stopping_patience: stop if val_iou does not improve for this many epochs
        early_stopping_min_delta: minimum improvement in val_iou to reset patience

    Returns:
        list[dict]: single-row results with the basic info
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_processes.txt")
    figures_dir = os.path.join(results_dir, "figures"); os.makedirs(figures_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    curves_path = os.path.join(figures_dir, "retinanet_training_curves.png")

    # Build model and optimizer
    model = build_retinanet(num_classes).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    model.to(device)

    # History for plots
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    # Early stopping state
    best_val_iou = -float("inf")
    best_state_dict = None
    patience_counter = 0
    best_epoch = 0

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n>>> Training of RetinaNet ...\n")
        f.write(f"Epochs: {epochs} | LR: {lr} | Momentum: {momentum} | Weight Decay: {weight_decay}\n")
        f.write(f"EarlyStopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}\n\n")

    # TRAIN and VALIDATE
    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device)

        # Validate (loss + val IoU)
        val_loss, val_iou = validate_one_epoch(model, val_loader, device)

        # model.eval()
        # iou_sum, n_imgs = 0.0, 0
        # with torch.no_grad():
        #     for images, targets in train_loader:
        #         images = [img.to(device) for img in images]
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #         outputs = model(images)
        #         for out, tgt in zip(outputs, targets):
        #             pred_boxes = out.get("boxes", torch.empty(0, 4, device=device))
        #             gt_boxes = tgt.get("boxes", torch.empty(0, 4, device=device))
        #             mean_iou, _, _ = iou_metrics(pred_boxes, gt_boxes, iou_thr=0.5)
        #             iou_sum += mean_iou
        #             n_imgs += 1
        # train_iou = iou_sum / max(1, n_imgs)
        train_iou = float("nan")

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        # Log line
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch: {epoch} | "
                f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f} | "
                f"Train IoU: {train_iou:.6f}  Val IoU: {val_iou:.6f}\n"
            )

        # Early Stopping on val_iou
        improved = (val_iou - best_val_iou) > early_stopping_min_delta
        if improved:
            best_val_iou = val_iou
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"Early stopping triggered at epoch {epoch} (best val_iou at epoch {best_epoch}: {best_val_iou:.6f})\n")
                break

    training_time_sec = time.time() - t0_train

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Count trainable params
    n_params = count_parameters(model)

    # Save training/validation curves
    plot_training_curves(history, save_path=curves_path)

    # TEST EVAL
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()
    total_iou, total_prec, total_rec, n_imgs = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
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
    model.eval()
    total_time, total_imgs = 0.0, 0
    with torch.no_grad():
        for images, *_ in test_loader:
            images = [img.to(device) for img in images]
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            _ = model(images)
            if device.type == "cuda": torch.cuda.synchronize()
            total_time += (time.time() - t0)
            total_imgs += len(images)
    avg_inf_time = total_time / max(1, total_imgs)

    # Log some info
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\nTEST IoU@0.5={iou05:.4f} Prec@0.5={precision05:.4f} Rec@0.5={recall05:.4f}\n")
        f.write(
            f"Training finished in {training_time_sec:.2f} seconds | "
            f"Trainable parameters: {n_params:,}\n"
        )

    # Save trained model
    torch.save(model.state_dict(), os.path.join(models_dir, "retinanet.pth"))

    row = {
        "Model": "RetinaNet",
        "IoU@0.5": iou05,
        "Precision@0.5": precision05,
        "Recall@0.5": recall05,
        "mAP@0.5": map05,
        "Training Time (s)": training_time_sec,
        "Average Inference Time (s)": avg_inf_time,
        "#Params": n_params,
    }
    return [row]
