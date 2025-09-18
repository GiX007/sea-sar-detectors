# src/predict_single.py
#
# Minimal single-image predictor for Faster R-CNN, RetinaNet, and YOLOv8.
# - Loads one image, prints [xyxy, score, label] and saves annotated image.
#
import os, json, argparse
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
from torchvision import transforms


# parsers
def parse_args():
    p = argparse.ArgumentParser("Predict on a single image")
    p.add_argument("--model_type", required=True, choices=["fasterrcnn", "retinanet", "yolov8"])
    p.add_argument("--model_path", required=True, help="Path to weights under models/")
    p.add_argument("--image_path", required=True, help="Path to image under test_images/")
    p.add_argument("--class_map", default="models/class_map.json", help="List of class names (no background)")
    p.add_argument("--score_thr", type=float, default=0.30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# utils
def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)  # list like ["swimmer","boat",...]
    if isinstance(obj, list):
        return {i: n for i, n in enumerate(obj)}  # 0..C-1
    raise ValueError("class_map.json must be a list of class names")


def draw_boxes(pil_img, boxes, scores, labels, id2name, thr=0.3):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    for b, s, l in zip(boxes, scores, labels):
        if s < thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        name = id2name.get(int(l), str(int(l)))
        txt = f"{name} {s:.2f}"
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

        # get text width/height
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # fallback for older Pillow
            tw, th = draw.textsize(txt, font=font)

        draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 4, y1)], fill=(255, 0, 0))
        draw.text((x1 + 2, y1 - th - 2), txt, fill=(255, 255, 255), font=font)

    return pil_img


def print_dets(boxes, scores, labels, id2name, thr):
    print(">>> Predictions (xyxy, score, label)")
    any_ok = False
    for b, s, l in zip(boxes, scores, labels):
        if s < thr:
            continue
        x1, y1, x2, y2 = [round(float(v), 2) for v in b]
        print(f"box={[x1,y1,x2,y2]}, score={s:.3f}, label={id2name.get(int(l), str(int(l)))} (id={int(l)})")
        any_ok = True
    if not any_ok:
        print(f"No detections above threshold {thr}")

def save_with_suffix(image_path, model_type, pil_img, thr):
    # all predictions under results/test_images_preds/
    out_dir = os.path.join("results", "test_images_preds")
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.basename(image_path)
    stem, ext = os.path.splitext(basename)
    out_name = f"{stem}__{model_type}__thr{str(thr)}__pred{ext}"
    out_path = os.path.join(out_dir, out_name)

    pil_img.save(out_path)


# model-specific inference
def load_tv_model(model_type, num_classes, device):
    if model_type == "fasterrcnn":
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_f, num_classes)
    elif model_type == "retinanet":
        from src.models.retinanet import build_retinanet
        m = build_retinanet(num_classes)
    else:
        raise ValueError("Unsupported torchvision model")
    m.to(device).eval()
    return m


@torch.inference_mode()
def infer_torchvision(model, image_path, device):
    pil = Image.open(image_path).convert("RGB")
    timg = transforms.ToTensor()(pil).to(device) # [0,1]
    out = model([timg])[0]
    boxes = out["boxes"].detach().cpu().tolist()
    scores = out["scores"].detach().cpu().tolist()
    labels = out["labels"].detach().cpu().tolist()
    return pil, boxes, scores, labels


def infer_yolov8(weights_path, image_path, thr):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    res = model.predict(source=image_path, conf=thr, verbose=False)[0]
    pil = Image.open(image_path).convert("RGB")
    boxes = res.boxes.xyxy.cpu().tolist()
    scores = res.boxes.conf.cpu().tolist()
    labels = res.boxes.cls.cpu().tolist()
    return pil, boxes, scores, labels


# main
def main():
    args = parse_args()
    id2name = load_class_map(args.class_map)

    if args.model_type in {"fasterrcnn", "retinanet"}:
        num_classes = len(id2name) + 1  # +background
        model = load_tv_model(args.model_type, num_classes, args.device)

        state = torch.load(args.model_path, map_location=args.device)
        sd = state.get("model_state", state.get("state_dict", state))
        model.load_state_dict(sd, strict=False)

        pil, boxes, scores, labels = infer_torchvision(model, args.image_path, args.device)
        labels = [max(0, int(l) - 1) for l in labels]  # shift (1..C) -> (0..C-1)

    elif args.model_type == "yolov8":
        pil, boxes, scores, labels = infer_yolov8(args.model_path, args.image_path, args.score_thr)
        labels = [int(l) for l in labels]  # already 0..C-1

    else:
        raise ValueError("Unsupported model_type")

    print_dets(boxes, scores, labels, id2name, args.score_thr)
    annotated = draw_boxes(pil.copy(), boxes, scores, labels, id2name, args.score_thr)
    save_with_suffix(args.image_path, args.model_type, annotated, args.score_thr)


if __name__ == "__main__":
    main()
