# src.eda.py
#
# Helper functions for Exploratory Data Analysis (EDA).
#
import os, random, json
import numpy as np
from collections import Counter
from PIL import Image
from src.utils import write_log, save_bboxes_torchvision


def list_folders_and_files(root_dir, log_file):
    """Logs what subfolders and files exist in the dataset root."""
    write_log(">>> Folder structure:", filename=log_file)
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            write_log(f"[DIR]  {item}", filename=log_file)
            for sub in os.listdir(path):
                write_log(f"    - {sub}", filename=log_file)
        else:
            write_log(f"[FILE] {item}", filename=log_file)
    write_log("", filename=log_file)


def count_files_by_extension(directory, log_file):
    """Counts files by extension in a directory tree and log results."""
    counts = Counter()
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            counts[ext] += 1

    write_log(f">>> File counts in {directory}:", filename=log_file)
    for ext, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        write_log(f"  {ext}: {cnt}", filename=log_file)
    write_log("", filename=log_file)
    return counts


def summarize_images(images_dir, log_file):
    """Summarizes dtype and shape across all images in a directory tree."""
    dtype_counts = Counter()
    shape_counts = Counter()
    total_images = 0

    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            total_images += 1
            path = os.path.join(root, f)
            with Image.open(path) as im:
                arr = np.array(im)
            dtype_counts[str(arr.dtype)] += 1
            if arr.ndim == 2:
                shape_key = f"{arr.shape[0]}x{arr.shape[1]}"
            elif arr.ndim == 3:
                shape_key = f"{arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]}"
            else:
                shape_key = "unknown_shape"
            shape_counts[shape_key] += 1

    write_log(f">>> Image summary in {images_dir}:", filename=log_file)
    write_log(f"  Total images: {total_images}", filename=log_file)
    write_log("  Dtype distribution:", filename=log_file)
    for k, v in sorted(dtype_counts.items(), key=lambda x: x[1], reverse=True):
        write_log(f"    {k}: {v}", filename=log_file)
    write_log("  Shape distribution:", filename=log_file)
    for k, v in sorted(shape_counts.items(), key=lambda x: x[1], reverse=True):
        write_log(f"    {k}: {v}", filename=log_file)
    write_log("", filename=log_file)


def explore_json(json_path, log_file):
    """Explores top-level keys and preview values in a COCO-style JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    write_log(f">>> Exploring JSON: {json_path}", filename=log_file)
    write_log("  Top-level keys: " + ", ".join(data.keys()), filename=log_file)

    # loop over keys in the JSON
    for k, v in data.items():
        if isinstance(v, list):
            write_log(f"  {k}: list of length {len(v)}", filename=log_file)
            if k == "categories":  # special handling for categories
                write_log("    Classes (id → name):", filename=log_file)
                for cat in v:
                    write_log(f"      {cat['id']}: {cat['name']}", filename=log_file)
            if len(v) > 0:
                write_log(f"    first item: {str(v[0])[:200]}", filename=log_file)

        elif isinstance(v, dict):
            keys_preview = ", ".join(list(v.keys()))
            write_log(f"  {k}: dict with {len(v)} keys", filename=log_file)
            write_log(f"    keys: {keys_preview}", filename=log_file)

        else:
            write_log(f"  {k}: {type(v).__name__}, value={v}", filename=log_file)

    write_log("", filename=log_file)


def pick_random_train_image(train_dir, train_json, log_file):
    """
    Picks a random image from the train set, logs metadata and annotations,
    saves a preview, and returns info for downstream use.

    Returns:
        tuple: (image_path, anns, cat_id_to_name)
            - image_path (str): Path to the chosen image
            - anns (list[dict]): List of annotation dicts for that image
            - cat_id_to_name (dict[int,str]): Maps category_id -> class name
    """
    # pick a random image file
    jpgs = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    fname = random.choice(jpgs)
    path = os.path.join(train_dir, fname)
    stem = os.path.splitext(fname)[0]

    # preview image save
    with Image.open(path) as im:
        arr = np.array(im)
        os.makedirs("results/figures", exist_ok=True)
        preview_path = os.path.join("results/figures", f"{stem}_preview_train.jpg")
        im.save(preview_path)

    # log image info
    write_log(">>> Random train sample:", filename=log_file)
    write_log(f"  Image name: {fname}", filename=log_file)
    write_log(f"  Image path: {path}", filename=log_file)
    write_log(f"  Dtype: {arr.dtype}", filename=log_file)
    write_log(f"  Shape: {arr.shape}", filename=log_file)
    # log full array (may be large!)
    write_log(f"  Image:\n {arr}", filename=log_file)

    # load JSON and annotations
    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    anns_by_img = {}
    for ann in data.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # category id → name mapping
    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}

    # find metadata and anns for this image
    img_info, anns = None, []
    for img in data.get("images", []):
        if img["file_name"] == fname:
            img_info = img
            anns = anns_by_img.get(img["id"], [])
            break

    if img_info:
        write_log(f"  Metadata   : {img_info}", filename=log_file)
        write_log(f"  Annotations: {anns}, total={len(anns)}", filename=log_file)
    else:
        write_log("  No metadata/annotations found in JSON.", filename=log_file)

    write_log("", filename=log_file)
    return path, anns, cat_id_to_name, stem


def run_eda(data_dir):
    """
    High-level EDA:
      - dataset structure
      - counts + image summary for train/val/test
      - explore annotation JSONs
      - pick random train image and log metadata + annotations
    """
    log_file = "results/data_summary.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # start the txt
    write_log("=== Dataset EDA Report ===", filename=log_file, mode="w")
    write_log(f"Root data dir: {data_dir}", filename=log_file)
    write_log("", filename=log_file)

    # 1. folder structure
    list_folders_and_files(data_dir, log_file)

    # 2. explore images (train/val/test)
    for split in ["train", "val", "test"]:
        images_dir = os.path.join(data_dir, "images", split)
        if os.path.exists(images_dir):
            count_files_by_extension(images_dir, log_file)
            summarize_images(images_dir, log_file)

    # 3. explore jsons
    ann_dir = os.path.join(data_dir, "annotations")
    for f in os.listdir(ann_dir):
        if f.endswith(".json"):
            explore_json(os.path.join(ann_dir, f), log_file)

    # 4. random train image and annotation
    train_dir = os.path.join(data_dir, "images", "train")
    train_json = os.path.join(data_dir, "annotations", "instances_train.json")
    if os.path.exists(train_dir) and os.path.exists(train_json):
        sample_path, anns, cat_id_to_name, stem = pick_random_train_image(train_dir, train_json, log_file)

        # save version with bboxes
        boxed_out = os.path.join("results", "figures", f"{stem}_train_bboxes.jpg")
        save_bboxes_torchvision(sample_path, anns, cat_id_to_name, boxed_out)
