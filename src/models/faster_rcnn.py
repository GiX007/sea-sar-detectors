# src/faster_rcnn.py
#
# Original Faster R-CNN paper: https://arxiv.org/abs/1506.01497
#
# Minimal Faster R-CNN runner built on TorchVision's finetuning tutorial:
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#
import os, time, torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.utils import plot_training_curves, count_parameters, train_one_epoch, validate_one_epoch, iou_metrics


def build_faster_rcnn(num_classes):
    """
    Builds a Faster R-CNN model (ResNet50-FPN) and replace its classifier head.

    Args:
        num_classes (int): Number of classes INCLUDING background

    Returns:
        torch.nn.Module: Faster R-CNN model ready to train
    """
    # Load a COCO-pretrained backbone
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", trainable_backbone_layers=0)

    # ↓ Cut proposal counts and per-image samples (big speed win on GTX 1050 Ti)
    model.rpn._pre_nms_top_n = {'training': 1000, 'testing': 600} # fewer candidates passed to NMS
    model.rpn._post_nms_top_n = {'training': 300, 'testing': 150} # fewer final proposals per image
    model.rpn.batch_size_per_image = 64 # was 256 (anchors sampled for the loss)
    model.roi_heads.batch_size_per_image = 64 # was 512 (RoIs sampled for the head)

    # Replace the classifier head to match dataset classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def run_faster_rcnn(
    train_loader, val_loader, test_loader, num_classes, device,
    epochs=100, lr=0.005, momentum=0.9, weight_decay=0.0005, results_dir="results",
    early_stopping_patience=10, early_stopping_min_delta=1e-4
):
    """
    Trains, validates, plots curves, and evaluates Faster R-CNN on the TEST set with early stopping on validation IoU@0.5.

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
        list[dict]: single-row results with basic info
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_processes.txt")
    figures_dir = os.path.join(results_dir, "figures"); os.makedirs(figures_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    curves_path = os.path.join(figures_dir, "faster_rcnn_training_curves.png")

    # Build model and optimizer
    model = build_faster_rcnn(num_classes).to(device)
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
        f.write(f"\n>>> Training of Faster R-CNN ...\n")
        f.write(f"Epochs: {epochs} | LR: {lr} | Momentum: {momentum} | Weight Decay: {weight_decay}\n")
        f.write(f"EarlyStopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}\n\n")

    # TRAIN and VALIDATE
    t0_train = time.time()
    for epoch in range(1, epochs + 1):

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device)

        # Validate (loss + val IoU)
        val_loss, val_iou = validate_one_epoch(model, val_loader, device)

        # Compute train IoU (eval mode, no grad)
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

            # GPU is asynchronous -> sync before/after to measure real compute time
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

            dt = time.time() - t0
            total_time += dt
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
    torch.save(model.state_dict(), os.path.join(models_dir, "fasterrcnn.pth"))

    # Result row
    row = {
        "Model": "Faster R-CNN",
        "IoU@0.5": iou05,
        "Precision@0.5": precision05,
        "Recall@0.5": recall05,
        "mAP@0.5": map05,
        "Training Time (s)": training_time_sec,
        "Average Inference Time (s)": avg_inf_time,
        "#Params": n_params,
    }
    return [row]
