# src/utils.py
#
# Helper functions for common operations.
#
import os, torch, datetime
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def write_log(message, filename="results/data_summary.txt", mode="a"):
    """
    Appends a message to a log file (creates a file if not exists).

    Args:
        message (str): The message to append
        filename (str): The filename to save the log to
        mode (str): The mode to save the log. It can be 'a' for appending or 'w' for writing to a clear file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # convert all to strings
    if not isinstance(message, str):
        message = str(message)

    with open(filename, mode, encoding="utf-8") as f:
        f.write(message + "\n")


def log_device_info(device, filename="results/training_processes.txt"):
    """
    Writes a short header about the training device/environment to the training log.

    Args:
        device (torch.device): The device used for training (cpu or cuda)
        filename (str): Log file path
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    write_log("=== Training Session ===", filename=filename, mode="a")
    write_log(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", filename=filename)

    write_log(f"PyTorch: {torch.__version__}", filename=filename)
    write_log(f"Device: {device.type}", filename=filename)

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        write_log(f"CUDA: available | GPU={name} | capability={cap[0]}.{cap[1]} | VRAM={mem:.1f} GB", filename=filename)
        write_log(f"cuDNN: {torch.backends.cudnn.version()}", filename=filename)
    else:
        write_log("CUDA: not available", filename=filename)


def save_bboxes_torchvision(image_path, anns, cat_id_to_name, out_path):
    """
    Draws COCO-style bboxes [x,y,w,h] + class labels above the boxes and save it to out_path using torchvision.
    """
    img = read_image(image_path) # load as tensor (C,H,W), uint8

    boxes, labels = [], []
    for a in anns:
        x, y, w, h = a["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        boxes.append([x1, y1, x2, y2]) # real box for rectangle
        labels.append(str(cat_id_to_name.get(a["category_id"])))

    if boxes:  # draw only if there are boxes
        boxes_t = torch.tensor(boxes, dtype=torch.int)
        img = draw_bounding_boxes(img, boxes_t, labels=labels, colors=["red"] * len(boxes), width=3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    to_pil_image(img).save(out_path)


def intersection_over_union(boxes_preds, boxes_target, box_format="midpoint", eps=1e-6):
    """
    Computes IoU (Intersection over Union) between predicted and target boxes.

    Args:
        boxes_preds (torch.Tensor): Predicted boxes (..., 4)
        boxes_target (torch.Tensor): Target boxes (..., 4), same format as boxes_preds
        box_format (str): "midpoint" or "corners". Default = "midpoint"
        eps (float): Small number to avoid division by zero. Default = 1e-6

    Returns:
        torch.Tensor: IoU values with shape (...) (same leading dims as input, the last dim removed)

    Notes:
        - "midpoint" format means (x_center, y_center, width, height), also called xywh
        - "corners" format means (x1, y1, x2, y2), also called xyxy, where (x1, y1) is top-left, and (x2, y2) is bottom-right
        - In our dataset, boxes come in "corners" (xyxy) format with absolute pixel values (e.g., [370.6, 212.7, 377.6, 226.9] for a 640×640 image)
    """
    if box_format == "midpoint":
        # Convert (x_center, y_center, width, height) → (x1, y1, x2, y2). We subtract/ add half width and height to move from center-size to corner format
        # /2 because converting from center-size to corners: x1 = xc - w/2, x2 = xc + w/2; same for y.
        px = boxes_preds[..., 0]; py = boxes_preds[..., 1]
        pw = boxes_preds[..., 2]; ph = boxes_preds[..., 3]
        p_x1 = px - pw / 2; p_y1 = py - ph / 2
        p_x2 = px + pw / 2; p_y2 = py + ph / 2

        tx = boxes_target[..., 0]; ty = boxes_target[..., 1]
        tw = boxes_target[..., 2]; th = boxes_target[..., 3]
        t_x1 = tx - tw / 2; t_y1 = ty - th / 2
        t_x2 = tx + tw / 2; t_y2 = ty + th / 2

    elif box_format == "corners":
        p_x1 = boxes_preds[..., 0]; p_y1 = boxes_preds[..., 1]
        p_x2 = boxes_preds[..., 2]; p_y2 = boxes_preds[..., 3]
        t_x1 = boxes_target[..., 0]; t_y1 = boxes_target[..., 1]
        t_x2 = boxes_target[..., 2]; t_y2 = boxes_target[..., 3]
    else:
        raise ValueError("box_format must be 'midpoint' or 'corners'")

    # Intersection corners
    x1 = torch.max(p_x1, t_x1)
    y1 = torch.max(p_y1, t_y1)
    x2 = torch.min(p_x2, t_x2)
    y2 = torch.min(p_y2, t_y2)

    # Clamp avoids negative sizes when boxes don't overlap
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)

    inter_area = inter_w * inter_h
    pred_area = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    targ_area = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)

    union = pred_area + targ_area - inter_area
    return inter_area / (union + eps)


def iou_metrics(pred_boxes, gt_boxes, iou_thr=0.5):
    """
    Computes mean IoU, precision, and recall using greedy matching at IoU >= iou_thr.

    Args:
        pred_boxes (Tensor): shape (P, 4), predicted boxes in xyxy format
        gt_boxes (Tensor): shape (G, 4), ground-truth boxes in xyxy format
        iou_thr (float): IoU threshold to consider a match valid

    Returns:
        tuple: (mean_iou, precision, recall)
    """
    # Handle corner cases
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        # No predictions and no ground truth → nothing to match
        # Convention: return perfect scores (1,1,1) to avoid division by zero
        return 1.0, 1.0, 1.0
    if pred_boxes.numel() == 0:
        # No predictions but GT exists → complete failure
        return 0.0, 0.0, 0.0
    if gt_boxes.numel() == 0:
        # Predictions but no GT → all are false positives
        return 0.0, 0.0, 1.0

    # Step 1. Compute IoUs

    # ious[p, g] = IoU between predicted box p and GT box g
    ious = box_iou(pred_boxes, gt_boxes)  # shape (P, G)

    # Get all (pred, gt) index pairs where IoU >= threshold
    pairs = (ious >= iou_thr).nonzero(as_tuple=False)
    if pairs.numel() == 0:
        # No prediction overlaps a GT (enough) → no matches
        return 0.0, 0.0, 0.0

    # Step 2. Sort pairs by IoU

    # vals = IoU values of candidate (pred, gt) pairs
    vals = ious[pairs[:, 0], pairs[:, 1]]
    # Sort in descending order so we try to match the highest IoU first
    order = torch.argsort(vals, descending=True)
    pairs = pairs[order]

    # Step 3. Greedy matching

    used_p, used_g = set(), set() # keep track of already-matched preds & GTs
    iou_list = [] # store IoU values of accepted matches

    for p, g in pairs:
        p, g = int(p.item()), int(g.item())
        # Skip if either this prediction or GT is already matched
        if p in used_p or g in used_g:
            continue
        # Otherwise, accept this match
        used_p.add(p)
        used_g.add(g)
        iou_list.append(float(ious[p, g].item()))

    # Step 4. Compute metrics

    precision = len(used_p) / max(pred_boxes.size(0), 1) # matched preds / all preds
    recall = len(used_g) / max(gt_boxes.size(0), 1) # matched GTs / all GTs
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0

    return mean_iou, precision, recall


def train_one_epoch(model, optimizer, train_loader, device):
    """
    Trains the detection model for one epoch using loss returned by torchvision models.

    Args:
        model: detection model (e.g., Faster R-CNN)
        optimizer: torch optimizer
        train_loader: dataloader yielding (images, targets)
        device: torch device

    Returns:
        float: average training loss for this epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward returns a dict of losses for detection models
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    return avg_loss


def validate_one_epoch(model, val_loader, device):
    """
    Validates the detection model for one epoch, computing loss and IoU@0.5

    Args:
        model: detection model (e.g., Faster R-CNN)
        val_loader: dataloader yielding (images, targets)
        device: torch device

    Returns:
        float: average validation loss
        float: average IoU at 0.5 over the dataset
    """
    # compute validation loss (use training mode to get loss dict, no grads)
    model.train()
    val_loss_sum, val_batches = 0.0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)  # returns dict of losses in train mode
            loss = sum(loss for loss in loss_dict.values())
            val_loss_sum += float(loss.item())
            val_batches += 1
    avg_val_loss = val_loss_sum / max(1, val_batches)

    # compute IoU@0.5 (eval mode returns predictions)
    model.eval()
    iou_sum, images_count = 0.0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)  # list of dicts with boxes, labels, scores
            for out, tgt in zip(outputs, targets):
                pred_boxes = out.get("boxes", torch.empty(0, 4, device=device))
                gt_boxes = tgt.get("boxes", torch.empty(0, 4, device=device))
                mean_iou, _, _ = iou_metrics(pred_boxes, gt_boxes, iou_thr=0.5)
                iou_sum += float(mean_iou)
                images_count += 1

    avg_val_iou = iou_sum / max(1, images_count)
    return avg_val_loss, avg_val_iou


def count_parameters(model):
    """
    Counts trainable parameters of a model

    Args:
        model: torch nn.Module

    Returns:
        int: number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(history, save_path="results/training_curves.png"):
    """
    Plots train/val loss and IoU curves and saves to disk.

    Args:
        history: dict with lists per epoch. Expected keys:
                 'train_loss', 'val_loss', 'val_iou' (optional 'train_iou')
        save_path: output image path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_iou = history.get("val_iou", [])
    train_iou = history.get("train_iou", [])

    epochs_loss = range(1, len(train_loss) + 1) if train_loss else range(1, len(val_loss) + 1)
    epochs_iou = range(1, len(val_iou) + 1)

    # Create a figure with two subplots: Loss and IoU
    plt.figure(figsize=(10, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    if train_loss:
        plt.plot(epochs_loss, train_loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs_loss, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    # IoU subplot
    plt.subplot(1, 2, 2)
    if train_iou:
        plt.plot(epochs_iou, train_iou, label="Train IoU")
    if val_iou:
        plt.plot(epochs_iou, val_iou, label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU@0.5")
    plt.title("IoU Curves")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
