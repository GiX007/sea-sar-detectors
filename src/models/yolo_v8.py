# src/models/yolo_v8.py
#
# Minimal YOLOv8 runner using Ultralytics.
#
# References from Ultralytics YOLOv8 docs: https://docs.ultralytics.com
#
import os, time, json, torch, contextlib, logging, shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from src.utils import iou_metrics


# Helpers
def _plot_yolo_curves(results_csv_path, save_path):
    """
    Plots training curves from Ultralytics results.csv (if present).
    We draw: train/val box_loss and a mAP50 line (column name varies by version).
    """
    if not results_csv_path or not os.path.exists(results_csv_path):
        return

    df = pd.read_csv(results_csv_path)
    if df.empty:
        return

    # x-axis
    x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))

    plt.figure()
    plotted = False

    # train/val loss (sum common components if present)
    t_cols = [c for c in ["train/box_loss", "train/cls_loss", "train/dfl_loss"] if c in df.columns]
    if t_cols:
        plt.plot(x, df[t_cols].sum(axis=1), label="train/loss")
        plotted = True

    v_cols = [c for c in ["val/box_loss", "val/cls_loss", "val/dfl_loss"] if c in df.columns]
    if v_cols:
        plt.plot(x, df[v_cols].sum(axis=1), label="val/loss")
        plotted = True

    # mAP50 (name varies by version)
    for cand in ["metrics/mAP50(B)", "metrics/mAP50", "val/mAP50(B)", "mAP50"]:
        if cand in df.columns:
            plt.plot(x, df[cand], label=cand)
            plotted = True
            break

    # If nothing was plotted, bail out silently
    if not plotted:
        plt.close()
        return

    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("YOLOv8 training curves")

    # Only add legend if there are labeled artists
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _predict_one_image_ultra(yolo_model, img_t):
    """
    Runs YOLOv8 on a single tensor image [C, H, W] in [0,1].

    Args:
        model: Ultralytics YOLO object
        img_t (Tensor): image in [0,1], shape [3, H, W]

    Returns:
        boxes_xyxy (Tensor [N,4]): predicted xyxy boxes on the model device
    """
    arr = (img_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # verbose=False avoids progress lines, any remaining prints is silenced by our redirects
    res = yolo_model.predict(arr, verbose=False)[0]
    dev = next(yolo_model.model.parameters()).device
    if res.boxes is None or len(res.boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=dev)
    return res.boxes.xyxy.to(dev)


def _ensure_yolo_labels(resplit_root: str):
    """Creates YOLO .txt labels from COCO json if they don't already exist."""
    ann_dir = Path(resplit_root) / "annotations"
    lbl_root = Path(resplit_root) / "labels"
    lbl_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        json_path = ann_dir / f"instances_{split}.json"
        out_dir = lbl_root / split
        if out_dir.exists() and any(out_dir.glob("*.txt")):
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        cats = [c for c in data["categories"] if c.get("id", -1) != 0 and c["name"].lower() != "ignored"]
        cats = sorted(cats, key=lambda c: c["id"])
        id2new = {c["id"]: i for i, c in enumerate(cats)}
        imgs = {im["id"]: im for im in data["images"]}

        anns_by_img = {}
        for a in data["annotations"]:
            if a.get("iscrowd", 0) == 0 and a["category_id"] in id2new:
                anns_by_img.setdefault(a["image_id"], []).append(a)

        for im in imgs.values():
            W, H = im["width"], im["height"]
            lines = []
            for a in anns_by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                cx, cy, ww, hh = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
                cls = id2new[a["category_id"]]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            (out_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")


def _ensure_data_yaml(resplit_root: str, class_names):
    """Writes a data.yaml with ABS paths (avoids path-join surprises on Windows)."""
    root = Path(resplit_root).resolve()
    yaml_path = root / "data.yaml"
    txt = (
        f"train: {(root / 'images' / 'train').as_posix()}\n"
        f"val:   {(root / 'images' / 'val').as_posix()}\n"
        f"test:  {(root / 'images' / 'test').as_posix()}\n\n"
        "names:\n" + "\n".join([f"  {i}: {n}" for i, n in enumerate(class_names)])
    )
    yaml_path.write_text(txt, encoding="utf-8")
    return yaml_path.as_posix()


def _sum_cols(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].sum(axis=1) if cols else pd.Series([np.nan] * len(df), index=df.index)


def _collect_epoch_lines(results_csv: str):
    """Yields (epoch, train_loss, val_loss) from Ultralytics results.csv (robust to column renames)."""
    if not results_csv or not os.path.exists(results_csv):
        return []
    df = pd.read_csv(results_csv)
    if df.empty:
        return []

    # synthesize epochs if needed
    if "epoch" in df.columns:
        epochs = df["epoch"].astype(int).tolist()
    else:
        epochs = list(range(len(df)))

    # sum common loss parts if present
    train_loss = _sum_cols(df, ["train/box_loss", "train/cls_loss", "train/dfl_loss"]).fillna(0.0)
    val_loss = _sum_cols(df, ["val/box_loss", "val/cls_loss", "val/dfl_loss"]).fillna(0.0)

    out = []
    for i in range(len(df)):
        e = int(epochs[i]) + 1  # show 1..N
        tr = float(train_loss.iloc[i])
        vl = float(val_loss.iloc[i])
        out.append((e, tr, vl))
    return out


def _find_latest(path_glob: str):
    """Returns the latest file path by mtime for a glob pattern, else ''."""
    paths = list(Path(".").glob(path_glob)) if not any(ch in path_glob for ch in "\\/*") else list(Path().glob(path_glob))
    if not paths:
        paths = list(Path().glob(path_glob))  # safety
    if not paths:
        return ""
    return str(max(paths, key=lambda p: p.stat().st_mtime))


# Runner
def run_yolo_v8(
    train_loader, val_loader, test_loader, num_classes,   # unused by YOLO (no background class)
    device, epochs = 100, imgsz = 256, results_dir = "results", model_name = "yolov8n.pt",
    resplit_root=None, class_names=None,
):
    """
    Trains YOLOv8 with Ultralytics and evaluate on TEST. Also computes IoU@0.5/Prec/Rec with iou_metrics.

    Returns:
        list[dict]: one row with
            Model, IoU@0.5, Precision@0.5, Recall@0.5, mAP@0.5, Training Time (s),
            Average Inference Time (s), #Params
    """
    assert resplit_root is not None and class_names is not None, "Need resplit_root and class_names"

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_processes.txt")
    fig_dir = os.path.join(results_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    curves_path = os.path.join(fig_dir, "yolov8_training_curves.png")

    # prepare YOLO data (labels + yaml)
    _ensure_yolo_labels(resplit_root)
    data_yaml_path = _ensure_data_yaml(resplit_root, class_names)

    # choose the device for Ultralytics
    ul_device = 0 if (isinstance(device, torch.device) and device.type == "cuda") else "cpu"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n>>> Training of YOLOv8 ...\n")
        f.write(f"Epochs: {epochs} | imgsz: {imgsz}\n")
        f.write(f"EarlyStopping: patience=5, min_delta=n/a (Ultralytics)\n\n")

    # silence Ultralytics logs to terminal
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.CRITICAL)
    except Exception:
        pass

    # build model
    yolo = YOLO(model_name)

    # run training with all outputs redirected away from the console
    t0 = time.time()
    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
        yolo.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            device=ul_device,
            project=results_dir,
            name="yolov8",  # results/yolov8/...
            exist_ok=True,
            verbose=False,
            batch=8,
            workers=4,
            patience=5
        )
    training_time_sec = time.time() - t0

    # find the latest run dir and its results.csv
    run_dir = str(Path(results_dir) / "yolov8")
    results_csv = ""
    # search under results/yolov8/** for the newest results.csv
    candidates = sorted(Path(run_dir).glob("**/results.csv"), key=lambda p: p.stat().st_mtime)
    if candidates:
        results_csv = str(candidates[-1])

    # write per-epoch lines into the same txt
    for e, tr, vl in _collect_epoch_lines(results_csv):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch: {e} | "
                f"Train Loss: {tr:.6f}  Val Loss: {vl:.6f} | "
                f"Train IoU: nan  Val IoU: nan\n"
            )

    # plot curves
    _plot_yolo_curves(results_csv, curves_path)

    # find the most recent best.pt and last.pt inside results/yolov8/**/weights/
    best_candidates = sorted(Path(run_dir).glob("**/weights/best*.pt"), key=lambda p: p.stat().st_mtime)
    last_candidates = sorted(Path(run_dir).glob("**/weights/last*.pt"), key=lambda p: p.stat().st_mtime)

    best_pt = str(best_candidates[-1]) if best_candidates else ""
    last_pt = str(last_candidates[-1]) if last_candidates else ""

    # copy to a common folder
    if best_pt:
        shutil.copy2(best_pt, os.path.join(models_dir, "yolov8_best.pt"))
    if last_pt:
        shutil.copy2(last_pt, os.path.join(models_dir, "yolov8_last.pt"))

    # choose the weights we'll use for evaluation and inference
    yolo_best = YOLO(best_pt) if best_pt else yolo

    # mAP@0.5 from Ultralytics on TEST (silenced)
    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
        metrics = yolo_best.val(data=data_yaml_path, split="test", imgsz=imgsz, device=ul_device, verbose=False)
    map05 = float(getattr(metrics.box, "map50", float("nan")))

    # manual TEST IoU/Prec/Rec@0.5 + average inference time
    model_device = device if isinstance(device, torch.device) else torch.device("cpu")
    yolo_best.model.to(model_device)
    yolo_best.model.eval()

    total_iou = total_prec = total_rec = 0.0
    n_imgs = 0
    total_time = 0.0
    total_count = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(model_device) for img in images]
            targets = [{k: v.to(model_device) for k, v in t.items()} for t in targets]
            for img, tgt in zip(images, targets):
                if model_device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                pred_boxes = _predict_one_image_ultra(yolo_best, img)
                if model_device.type == "cuda":
                    torch.cuda.synchronize()
                total_time += (time.time() - t1)
                total_count += 1

                gt_boxes = tgt.get("boxes", torch.empty(0, 4, device=model_device))
                mean_iou, precision, recall = iou_metrics(pred_boxes, gt_boxes, iou_thr=0.5)
                total_iou += float(mean_iou)
                total_prec += float(precision)
                total_rec += float(recall)
                n_imgs += 1

    iou05 = total_iou / max(1, n_imgs)
    precision05 = total_prec / max(1, n_imgs)
    recall05 = total_rec / max(1, n_imgs)
    avg_inf_time = total_time / max(1, total_count)

    n_params = sum(p.numel() for p in yolo_best.model.parameters())

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\nTEST IoU@0.5={iou05:.4f} Prec@0.5={precision05:.4f} Rec@0.5={recall05:.4f}\n")
        f.write(
            f"Training finished in {training_time_sec:.2f} seconds | "
            f"Trainable parameters: {n_params:,}\n"
        )

    row = {
        "Model": "YOLOv8",
        "IoU@0.5": iou05,
        "Precision@0.5": precision05,
        "Recall@0.5": recall05,
        "mAP@0.5": map05,
        "Training Time (s)": training_time_sec,
        "Average Inference Time (s)": avg_inf_time,
        "#Params": n_params,
    }
    return [row]
