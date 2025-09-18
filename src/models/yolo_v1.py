# src/models/yolo_v1.py
#
# Original paper: https://arxiv.org/abs/1506.02640
#
# YOLOv1 baseline (B=2) with minimal code.
#
import os, time, torch
import torch.nn as nn
from src.utils import plot_training_curves, iou_metrics, count_parameters


# Model
class CNNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class Yolov1(nn.Module):
    """Compact YOLOv1: reduced repeats and tiny head. Output per image: (S*S*(C + 2*5)) with B=2."""
    def __init__(self, S=7, B=2, C=20, in_ch=3):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        # Small Darknet-ish backbone
        layers = []
        cfg = [
            (7,64,2,3), 'M',
            (3,192,1,1), 'M',
            (1,128,1,0), (3,256,1,1),
            (1,256,1,0), (3,512,1,1), 'M',
            # repeat block reduced from 4 -> 2
            (1,256,1,0), (3,512,1,1),
            (1,256,1,0), (3,512,1,1),
            (1,512,1,0), (3,1024,1,1), 'M',
            (1,512,1,0), (3,1024,1,1),
            (3,1024,1,1)
        ]
        c = in_ch
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(2,2))
            else:
                k, co, s, p = x
                layers.append(CNNBlock(c, co, k, s, p))
                c = co
        self.backbone = nn.Sequential(*layers)

        # Tiny head: GAP -> Linear to (S*S*(C+10))  (B=2 -> 10)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, S*S*(C + 10))

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).flatten(1) # (N, C)
        x = self.head(x) # (N, S*S*(C+10))
        return x


# Inline targets/decoding
def _make_targets_inline(targets, S, C, img_size):
    """
    Builds compact per-cell target (one GT per cell):
      (N, S, S, C+5) = [onehot(C in 0..K-1), obj, x,y,w,h] with x,y,w,h in [0,1].
    NOTE: YOLOv1 expects class indices in 0..K-1 (no background class here).
    """
    N = len(targets)
    tgt = torch.zeros((N, S, S, C+5), device=targets[0]["boxes"].device, dtype=torch.float32)

    cell = img_size / S
    for n, t in enumerate(targets):
        boxes = t["boxes"] # xyxy pixels
        labels = t["labels"] # TorchVision-style labels in 1..K (bg is 0 but never used)
        for k in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[k]
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)

            i = int(torch.clamp((yc / cell).floor(), 0, S - 1).item())
            j = int(torch.clamp((xc / cell).floor(), 0, S - 1).item())

            # if already occupied -> skip
            if tgt[n, i, j, C] == 1:
                continue

            # shift 1..K -> 0..K-1 for YOLOv1 one-hot
            cls = int(labels[k].item()) - 1
            if 0 <= cls < C:
                tgt[n, i, j, cls] = 1.0
            else:
                continue # label outside range (e.g., background)

            tgt[n, i, j, C] = 1.0 # objectness
            tgt[n, i, j, C+1:C+5] = torch.tensor(
                [xc / img_size, yc / img_size, w / img_size, h / img_size],
                device=tgt.device, dtype=tgt.dtype
            )
    return tgt


def _decode_inline(pred, S, B, C, img_size, conf_thresh):
    """
    Decodes a single image prediction tensor (S*S*(C+10)) -> dict of boxes/scores/labels.
    No NMS (tiny baseline). One or two boxes per cell if objectness over the threshold.
    """
    grid = pred.reshape(S, S, C + 5*B)
    # class score: argmax over C, we’ll attach this label to any active box in the cell
    cls_ids = torch.argmax(grid[..., :C], dim=-1)  # (S,S)

    boxes, scores, labels = [], [], []
    for i in range(S):
        for j in range(S):
            for b in range(B):
                obj  = grid[i,j,C + 5*b + 0]
                if obj < conf_thresh:
                    continue
                x = grid[i,j,C + 5*b + 1] * img_size
                y = grid[i,j,C + 5*b + 2] * img_size
                w = grid[i,j,C + 5*b + 3] * img_size
                h = grid[i,j,C + 5*b + 4] * img_size
                x1,y1 = x - w/2, y - h/2
                x2,y2 = x + w/2, y + h/2
                boxes.append(torch.tensor([x1,y1,x2,y2], device=pred.device))
                scores.append(obj)
                labels.append(int(cls_ids[i,j].item()))
    if len(boxes)==0:
        return {"boxes": torch.zeros((0,4), device=pred.device),
                "scores": torch.zeros((0,), device=pred.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=pred.device)}
    return {"boxes": torch.stack(boxes,0),
            "scores": torch.stack(scores,0),
            "labels": torch.tensor(labels, dtype=torch.long, device=pred.device)}


# YOLOv1 Loss (B=2, classic)
class YoloV1Loss(nn.Module):
    """
    MSE-based YOLOv1 loss with responsibility assignment:
      - target per cell is one GT (obj=1) or none (obj=0)
      - for obj=1: pick responsible box (higher IoU), coord+obj loss for it, noobj loss for the other box
      - for obj=0: noobj loss for both boxes
      - class loss when obj=1
    """
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.mse = nn.MSELoss(reduction="sum")
        self.lc, self.ln = lambda_coord, lambda_noobj

    @staticmethod
    def _xywh_to_xyxy(xywh, img_size=1.0):
        # xywh: (..., 4) with values in [0,1] if normalized
        # Avoid in-place ops on views from unbind/slicing
        x = xywh[..., 0:1] * img_size
        y = xywh[..., 1:1 + 1] * img_size
        w = xywh[..., 2:2 + 1] * img_size
        h = xywh[..., 3:3 + 1] * img_size
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return torch.cat([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _bbox_iou_xyxy(a, b, eps=1e-6):
        # a: (...,4), b: (...,4)
        x1 = torch.max(a[...,0], b[...,0])
        y1 = torch.max(a[...,1], b[...,1])
        x2 = torch.min(a[...,2], b[...,2])
        y2 = torch.min(a[...,3], b[...,3])
        iw = (x2 - x1).clamp(min=0)
        ih = (y2 - y1).clamp(min=0)
        inter = iw*ih
        area_a = (a[...,2]-a[...,0]).clamp(min=0) * (a[...,3]-a[...,1]).clamp(min=0)
        area_b = (b[...,2]-b[...,0]).clamp(min=0) * (b[...,3]-b[...,1]).clamp(min=0)
        union = area_a + area_b - inter + eps
        return inter / union

    def forward(self, pred, target):
        """pred: (N, S*S*(C+10)) and target: (N, S, S, C+5)"""
        N, S, B, C = pred.size(0), self.S, self.B, self.C
        pred = pred.view(N, S, S, C + 5*B)

        # split
        pred_cls = pred[...,:C] # (N,S,S,C)
        # box k view: (N,S,S,5) -> [obj, x, y, w, h]
        box = [pred[..., C+5*k:C+5*(k+1)] for k in range(B)]
        tgt_cls = target[...,:C] # (N,S,S,C)
        tgt_obj = target[...,C:C+1] # (N,S,S,1)
        tgt_xywh= target[...,C+1:C+5] # (N,S,S,4)

        # choose responsible box where obj=1 via IoU
        iou = []
        tgt_xyxy = self._xywh_to_xyxy(tgt_xywh, img_size=1.0)  # target normalized
        for k in range(B):
            box_xyxy = self._xywh_to_xyxy(box[k][...,1:5], img_size=1.0)
            iou.append(self._bbox_iou_xyxy(box_xyxy, tgt_xyxy))   # (N,S,S)
        iou = torch.stack(iou, dim=-1)  # (N,S,S,B)
        best = torch.argmax(iou, dim=-1, keepdim=True)  # (N,S,S,1) in {0,1}

        # masks
        obj_mask = tgt_obj # (N, S, S, 1) 1 where the object exists
        noobj_mask = 1.0 - obj_mask
        resp_mask = torch.zeros_like(iou) # (N,S,S,B)
        resp_mask.scatter_(-1, best, 1.0)
        resp_mask = resp_mask.unsqueeze(-1) # (N,S,S,B,1)

        # losses
        # class (only where obj=1)
        class_loss = self.mse(
            (obj_mask * pred_cls).reshape(N,-1),
            (obj_mask * tgt_cls ).reshape(N,-1)
        )

        # objectness: responsible box to 1, non-responsible (when obj=1) and all (when obj=0) to 0
        obj_losses = 0.0
        for k in range(B):
            pk_obj = box[k][...,0:1]  # (N,S,S,1)
            resp_k = resp_mask[...,k,:]  # (N,S,S,1)
            # responsible (obj=1): to 1
            obj_losses += self.mse((obj_mask*resp_k*pk_obj).reshape(N,-1),
                                   (obj_mask*resp_k*torch.ones_like(pk_obj)).reshape(N,-1))
            # other box when obj=1 -> noobj
            obj_losses += self.ln * self.mse((obj_mask*(1-resp_k)*pk_obj).reshape(N,-1),
                                             torch.zeros((N, obj_mask.numel()//N), device=pk_obj.device))
            # when no object -> both boxes noobj
            obj_losses += self.ln * self.mse((noobj_mask*pk_obj).reshape(N,-1),
                                             torch.zeros((N, noobj_mask.numel()//N), device=pk_obj.device))

        # coordinates (only responsible box, obj=1), sqrt on w,h as in paper
        coord_loss = 0.0
        for k in range(B):
            resp_k = resp_mask[...,k,:]
            pk_xywh = box[k][...,1:5]
            pk_xy = pk_xywh[...,:2]
            pk_wh = pk_xywh[...,2:]
            tk_xy = tgt_xywh[...,:2]
            tk_wh = tgt_xywh[...,2:]
            # sqrt transform on widths/heights
            pk_wh_s = torch.sign(pk_wh) * torch.sqrt(torch.clamp(pk_wh.abs(), min=1e-6))
            tk_wh_s = torch.sqrt(torch.clamp(tk_wh,     min=1e-6))
            coord_loss += self.mse((obj_mask*resp_k*pk_xy).reshape(N,-1),
                                   (obj_mask*resp_k*tk_xy).reshape(N,-1))
            coord_loss += self.mse((obj_mask*resp_k*pk_wh_s).reshape(N,-1),
                                   (obj_mask*resp_k*tk_wh_s).reshape(N,-1))
        coord_loss = self.lc * coord_loss

        loss = coord_loss + class_loss + obj_losses
        # normalize by batch size for stability
        return loss / max(N,1)

# Runner
def run_yolo_v1(
    train_loader, val_loader, test_loader, num_classes, device,
    S=7, B=2, img_size=256, epochs=100, lr=1e-3, conf_thresh=0.25,
    results_dir="results", early_stopping_patience=10, early_stopping_min_delta=1e-4
):
    """
        Trains/validates YOLOv1 baseline, saves curves, and evaluates on the TEST set with early stopping on validation IoU@0.5.

        Args:
            train_loader: training dataloader yielding (images, targets)
            val_loader: validation dataloader yielding (images, targets)
            test_loader: test dataloader yielding (images, targets)
            num_classes: number of classes INCLUDING background
            device: torch device ('cuda' or 'cpu')
            S: Split per image
            B: Boxes per image
            img_size: size of input image
            epochs: number of training epochs
            lr: learning rate
            conf_thresh: confidence threshold
            results_dir: directory to save logs/plots
            early_stopping_patience: stop if val_iou does not improve for this many epochs
            early_stopping_min_delta: minimum improvement in val_iou to reset patience

        Returns:
            list[dict]: single-row results with basic info
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_processes.txt")
    fig_dir = os.path.join(results_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    curves_path = os.path.join(fig_dir, "yolov1_training_curves.png")

    K = num_classes - 1
    model = Yolov1(S=S, B=B, C=K).to(device)
    criterion = YoloV1Loss(S=S, B=B, C=K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    best_val_iou = -float("inf")
    best_state, patience = None, 0
    best_epoch = 0

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n>>> Training of YOLOv1 ...\n")
        f.write(f"Epochs: {epochs} | LR: {lr}\n")
        f.write(f"EarlyStopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}\n\n")

    t0_train = time.time()
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        tr_loss, tr_seen = 0.0, 0
        for images, targets in train_loader:
            images = torch.stack(images, 0).to(device)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            tgt_grid = _make_targets_inline(targets, S=S, C=K, img_size=img_size)

            preds = model(images)
            loss = criterion(preds, tgt_grid)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            tr_loss += float(loss.item()) * bs
            tr_seen += bs
        train_loss = tr_loss / max(tr_seen,1)

        # Val loss + Val IoU
        model.eval()
        va_loss, va_seen = 0.0, 0
        val_iou_sum, val_imgs = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = torch.stack(images, 0).to(device)
                targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                tgt_grid = _make_targets_inline(targets, S=S, C=K, img_size=img_size)

                preds = model(images)
                loss = criterion(preds, tgt_grid)

                bs = images.size(0)
                va_loss += float(loss.item()) * bs
                va_seen += bs

                # decode & IoU
                for n in range(bs):
                    decoded = _decode_inline(preds[n], S, B, K, img_size, conf_thresh)
                    mean_iou, _, _ = iou_metrics(decoded["boxes"], targets[n]["boxes"], iou_thr=0.5)
                    val_iou_sum += float(mean_iou); val_imgs += 1

        val_loss = va_loss / max(va_seen,1)
        val_iou = val_iou_sum / max(val_imgs,1)

        # Train IoU (full train loader, eval mode, no grad)
        # train_iou_sum, train_imgs = 0.0, 0
        # with torch.no_grad():
        #     for images, targets in train_loader:
        #         images = torch.stack(images, 0).to(device)
        #         targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        #         preds = model(images)
        #         for n in range(images.size(0)):
        #             decoded = _decode_inline(preds[n], S, B, K, img_size, conf_thresh)
        #             mi, _, _ = iou_metrics(decoded["boxes"], targets[n]["boxes"], iou_thr=0.5)
        #             train_iou_sum += float(mi); train_imgs += 1
        # train_iou = train_iou_sum / max(train_imgs, 1)
        train_iou = float("nan")

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        # Per-epoch log
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
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
            best_epoch = epoch
        else:
            patience += 1
            if patience >= early_stopping_patience:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"Early stopping at epoch {epoch} (best val_iou at {best_epoch}: {best_val_iou:.6f})\n")
                break

    training_time_sec = time.time() - t0_train
    if best_state is not None:
        model.load_state_dict(best_state)

    # Params
    n_params = count_parameters(model)

    # curves
    plot_training_curves(history, save_path=curves_path)

    # TEST (IoU/Prec/Rec) + avg inference time
    model.eval()
    total_iou = total_p = total_r = 0.0
    n_imgs, total_time, total_im = 0, 0.0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            images  = torch.stack(images, 0).to(device)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            preds = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += (time.time() - t0)
            total_im += images.size(0)

            for n in range(images.size(0)):
                decoded = _decode_inline(preds[n], S, B, K, img_size, conf_thresh)
                mi, pr, rc = iou_metrics(decoded["boxes"], targets[n]["boxes"], iou_thr=0.5)
                total_iou += float(mi); total_p += float(pr); total_r += float(rc); n_imgs += 1

    iou05 = total_iou / max(1,n_imgs)
    precision05 = total_p / max(1,n_imgs)
    recall05 = total_r / max(1,n_imgs)
    avg_inf_time = total_time / max(1,total_im)

    # Log training info
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"TEST IoU@0.5={iou05:.4f} Prec@0.5={precision05:.4f} Rec@0.5={recall05:.4f}\n")
        f.write(
            f"Training finished in {training_time_sec:.2f} seconds | "
            f"Trainable parameters: {n_params:,}\n"
        )

    # Save trained model
    torch.save(model.state_dict(), os.path.join(models_dir, "yolov1.pth"))

    return [{
        "Model": "YOLOv1",
        "IoU@0.5": iou05,
        "Precision@0.5": precision05,
        "Recall@0.5": recall05,
        "mAP@0.5": float("nan"),  # NaN for this baseline
        "Training Time (s)": training_time_sec,
        "Average Inference Time (s)": avg_inf_time,
        "#Params": n_params,
    }]
