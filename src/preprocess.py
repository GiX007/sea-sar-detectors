# src/preprocess.py
#
# Preprocessing:
#   - Resplit COCO (merge train+val annotated pool -> new train/val/test by original ratios)
#   - Copy images + write new COCO JSONs under data/<out_dir_name>/
#   - Log split sizes + class distributions to results/data_summary.txt
#   - Build PyTorch Datasets and DataLoaders (uniform resize + bbox scaling)
#
import os, json, random, shutil, torch
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from src.utils import write_log


# basic file & json helpers

def list_jpgs(dir_path):
    """Returns all .jpg file names in a directory."""
    return [f for f in os.listdir(dir_path) if f.lower().endswith(".jpg")]

def read_counts(data_dir):
    """Counts how many images are in train, val, and test folders."""
    t = list_jpgs(os.path.join(data_dir, "images", "train"))
    v = list_jpgs(os.path.join(data_dir, "images", "val"))
    te = list_jpgs(os.path.join(data_dir, "images", "test"))
    return len(t), len(v), len(te)

def load_coco(json_path):
    """Loads a COCO-style JSON file into a dictionary."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_coco(train_json_path, val_json_path):
    """Merges train and val COCO JSONs into one dictionary."""
    dtr = load_coco(train_json_path)
    dvl = load_coco(val_json_path)
    merged = {
        "info": dtr.get("info", {}),
        "licenses": dtr.get("licenses", []),
        "categories": dtr.get("categories", []),
        "images": dtr.get("images", []) + dvl.get("images", []),
        "annotations": dtr.get("annotations", []) + dvl.get("annotations", []),
    }
    return merged

def compute_targets(n_total, r_tr, r_va):
    """Computes how many samples go to train/val/test given ratios."""
    n_tr = round(r_tr * n_total)
    n_va = round(r_va * n_total)
    n_te = n_total - n_tr - n_va
    return n_tr, n_va, n_te

def write_coco_subset(base_coco, img_list, out_json_path):
    """Writes a COCO JSON subset containing only selected images and their annotations."""
    ids = set([img["id"] for img in img_list])
    anns = [a for a in base_coco["annotations"] if a.get("image_id") in ids]
    out = {
        "info": base_coco.get("info", {}),
        "licenses": base_coco.get("licenses", []),
        "categories": base_coco.get("categories", []),
        "images": img_list,
        "annotations": anns,
    }
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

def copy_images(img_list, source_train_dir, source_val_dir, dest_split_dir):
    """Copies images from original train/val dirs to the new split dir."""
    os.makedirs(dest_split_dir, exist_ok=True)
    for img in img_list:
        fname = img.get("file_name")
        src = os.path.join(source_train_dir, fname)
        if not os.path.exists(src):
            src = os.path.join(source_val_dir, fname)
        dst = os.path.join(dest_split_dir, fname)
        shutil.copy2(src, dst)

def class_distribution(annotations, categories):
    """Counts how many annotations exist per class name."""
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cnt = Counter()
    for a in annotations:
        name = cat_id_to_name.get(a.get("category_id"), str(a.get("category_id")))
        cnt[name] += 1
    return dict(sorted(cnt.items(), key=lambda x: x[1], reverse=True))

def log_class_distributions(coco_json_path, title, log_file):
    """Logs class distribution (counts per class) to a result file."""
    data = load_coco(coco_json_path)
    dist = class_distribution(data.get("annotations", []), data.get("categories", []))
    write_log(f"{title} class distribution:", filename=log_file)
    if dist:
        for k, v in dist.items():
            write_log(f"  {k}: {v}", filename=log_file)
    else:
        write_log("  <no annotations>", filename=log_file)
    write_log("", filename=log_file)


# resplit

def create_resplit_matching_ratios(data_dir, seed=42, out_dir_name="resplit"):
    """
    Creates a new train/val/test split from the annotated pool (train+val), using original ratios computed from data/images/{train,val,test}.
    Writes new images and COCO JSONs under data/<out_dir_name>/... Logs sizes and class distributions.
    """
    random.seed(seed)
    log_file = "results/data_summary.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # original counts and ratios
    n_tr0, n_va0, n_te0 = read_counts(data_dir)
    total0 = n_tr0 + n_va0 + n_te0
    r_tr = n_tr0 / total0
    r_va = n_va0 / total0
    r_te = n_te0 / total0

    # merge annotated pool
    train_json = os.path.join(data_dir, "annotations", "instances_train.json")
    val_json = os.path.join(data_dir, "annotations", "instances_val.json")
    merged = merge_coco(train_json, val_json)
    annotated_images = merged.get("images", [])

    # shuffle + split by ratios
    random.shuffle(annotated_images)
    N = len(annotated_images)
    n_tr, n_va, n_te = compute_targets(N, r_tr, r_va)

    new_train_imgs = annotated_images[:n_tr]
    new_val_imgs = annotated_images[n_tr:n_tr+n_va]
    new_test_imgs = annotated_images[n_tr+n_va:]

    # write JSONs
    out_root = os.path.join(data_dir, out_dir_name)
    out_ann_dir = os.path.join(out_root, "annotations")
    write_coco_subset(merged, new_train_imgs, os.path.join(out_ann_dir, "instances_train.json"))
    write_coco_subset(merged, new_val_imgs, os.path.join(out_ann_dir, "instances_val.json"))
    write_coco_subset(merged, new_test_imgs, os.path.join(out_ann_dir, "instances_test.json"))

    # log classes (from categories, drop "ignored")
    cats = [c for c in merged.get("categories", [])
            if c.get("id") != 0 and c.get("name", "").lower() != "ignored"]
    class_names = [c["name"] for c in cats]
    write_log(f"Classes ({len(class_names)}) -> {class_names}", filename=log_file)

    # save models / class_map.json
    os.makedirs("models", exist_ok=True)
    with open("models/class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # copy images
    src_train_dir = os.path.join(data_dir, "images", "train")
    src_val_dir = os.path.join(data_dir, "images", "val")
    dst_train_dir = os.path.join(out_root, "images", "train")
    dst_val_dir = os.path.join(out_root, "images", "val")
    dst_test_dir = os.path.join(out_root, "images", "test")
    copy_images(new_train_imgs, src_train_dir, src_val_dir, dst_train_dir)
    copy_images(new_val_imgs, src_train_dir, src_val_dir, dst_val_dir)
    copy_images(new_test_imgs, src_train_dir, src_val_dir, dst_test_dir)

    # logs
    write_log("=== Resplit ===", filename=log_file)
    write_log(f"Original ratios (by images): train={r_tr:.4f}, val={r_va:.4f}, test={r_te:.4f}", filename=log_file)
    write_log(f"Annotated pool: {N}", filename=log_file)
    write_log(f"New split sizes -> train={len(new_train_imgs)}, val={len(new_val_imgs)}, test={len(new_test_imgs)}", filename=log_file)
    write_log(f"Output under: {out_root}", filename=log_file)
    write_log("", filename=log_file)

    log_class_distributions(os.path.join(out_ann_dir, "instances_train.json"), "[Resplit][train]", log_file)
    log_class_distributions(os.path.join(out_ann_dir, "instances_val.json"),   "[Resplit][val]",   log_file)
    log_class_distributions(os.path.join(out_ann_dir, "instances_test.json"),  "[Resplit][test]",  log_file)

    return out_root  # path to resplit root


# dataset / dataloaders

def build_class_map(instances_json_path):
    """
    Builds mappings from COCO category ids to contiguous indices suitable for TorchVision detection models.

    TorchVision convention:
      - Class index 0 is reserved for the background
      - Foreground classes must start at index 1

    Args:
        instances_json_path (str): Path to a COCO-style instances_*.json file (usually the training split).

    Returns:
        tuple:
            - class_id_to_idx (dict[int, int]): maps original COCO category_id -> contiguous index [1..C]
            - idx_to_name (list[str]): class names aligned to indices [0..C], where index 0 = "__background__"
            - num_classes (int): total number of classes including background (C + 1)
    """
    data = load_coco(instances_json_path)
    cats_all = sorted(data.get("categories", []), key=lambda c: c["id"])

    # drop the "ignored" category (id==0 or name=="ignored")
    cats = [c for c in cats_all if c.get("id", -1) != 0 and c.get("name", "").lower() != "ignored"]

    class_names = [c["name"] for c in cats]  # K names (no background)
    class_id_to_idx = {c["id"]: i + 1 for i, c in enumerate(cats)}  # 1..K
    idx_to_name = ["__background__"] + class_names  # 0 = background
    num_classes = len(idx_to_name)  # K + 1

    return class_id_to_idx, idx_to_name, num_classes


class SeaDrones_v2_Dataset(Dataset):
    """
    Minimal COCO-style detection dataset:
      returns (image_tensor, target_dict)
      - image: FloatTensor [C,H,W] in [0,1]
      - target: dict with keys: boxes (FloatTensor [N,4] xyxy), labels (LongTensor [N])
    Resizes image to (image_size) and scales bboxes accordingly.
    """
    def __init__(self, images_dir, instances_json_path, image_size=(448, 448), class_id_to_idx=None):
        self.images_dir = images_dir
        self.data = load_coco(instances_json_path)
        self.image_recs = self.data.get("images", [])
        self.anns_by_imgid = {}
        for a in self.data.get("annotations", []):
            self.anns_by_imgid.setdefault(a["image_id"], []).append(a)

        # mapping original COCO category_id -> contiguous index 0..C-1
        self.class_id_to_idx = class_id_to_idx

        self.image_size = image_size  # (H, W)
        self.tf = T.Compose([
            T.Resize(self.image_size, antialias=True),
            T.ToTensor(),  # [0,1], float32
        ])

    def __len__(self):
        return len(self.image_recs)

    def __getitem__(self, idx):
        rec = self.image_recs[idx]
        fname = rec["file_name"]
        img_id = rec["id"]
        path = os.path.join(self.images_dir, fname)

        # image
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size  # PIL: (W, H)
        img_resized = self.tf(img)  # torch [C, H, W] float32
        H, W = self.image_size

        # annotations -> scale boxes (COCO xywh -> xyxy)
        anns = self.anns_by_imgid.get(img_id, [])
        boxes = []
        labels = []
        sx = W / float(w0)
        sy = H / float(h0)
        for a in anns:
            cid = int(a["category_id"])
            # skip categories not in the map (e.g., "ignored")
            if self.class_id_to_idx is not None and cid not in self.class_id_to_idx:
                continue

            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue  # drop degenerate boxes

            x1 = x * sx
            y1 = y * sy
            x2 = (x + bw) * sx
            y2 = (y + bh) * sy
            boxes.append([x1, y1, x2, y2])

            if self.class_id_to_idx is not None:
                labels.append(int(self.class_id_to_idx[cid]))  # contiguous 0..C-1
            else:
                labels.append(cid)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }
        return img_resized, target


def collate_fn(batch):
    """
    Custom collate function for object detection DataLoader.

    Why we need this:
        - In detection, each image can have a different number of objects (bounding boxes).
        - PyTorch's default collate tries to stack targets into a single tensor → this fails
          because boxes/labels have different lengths.
        - Solution: keep images and targets as separate lists.

    Args:
        batch: list of (image, target) pairs from the Dataset

    Returns:
        images: list of image tensors (shape [3, H, W], float32)
        targets: list of dicts (each dict has keys "boxes", "labels", "image_id")
    """
    # Unzips a list of tuples [(img1,tgt1), (img2,tgt2), ...]
    # into two tuples: (img1,img2,...), (tgt1,tgt2,...)
    images, targets = zip(*batch)

    # Convert tuples back to lists (Faster R-CNN expects lists)
    return list(images), list(targets)


def _take_subset_indices(n, subset, seed=42):
    """
    Return a list of indices for a subset.
    - subset: None (use all), float in (0,1], or int (count).
    """
    if subset is None:
        return list(range(n))
    if isinstance(subset, float):
        k = max(1, min(n, int(round(subset * n))))
    else:
        k = max(1, min(n, int(subset)))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:k]
    return idx.tolist()


# for working with the full dataset
# def build_datasets_and_dataloaders(resplit_root, image_size=(192, 192), batch_size=4, num_workers=8):
#     """
#     Builds dataset objects and dataloaders for train/val/test splits.
#     Returns:
#         tuple: (train_loader, val_loader, test_loader, num_classes, idx_to_name)
#     """
#     ann_dir = os.path.join(resplit_root, "annotations")
#
#     # Build a class map from train JSON (consistent with training)
#     class_id_to_idx, idx_to_name, num_classes = build_class_map(
#         os.path.join(ann_dir, "instances_train.json")
#     )
#
#     train_ds = SeaDrones_v2_Dataset(
#         images_dir=os.path.join(resplit_root, "images", "train"),
#         instances_json_path=os.path.join(ann_dir, "instances_train.json"),
#         image_size=image_size,
#         class_id_to_idx=class_id_to_idx,
#     )
#     val_ds = SeaDrones_v2_Dataset(
#         images_dir=os.path.join(resplit_root, "images", "val"),
#         instances_json_path=os.path.join(ann_dir, "instances_val.json"),
#         image_size=image_size,
#         class_id_to_idx=class_id_to_idx,
#     )
#     test_ds = SeaDrones_v2_Dataset(
#         images_dir=os.path.join(resplit_root, "images", "test"),
#         instances_json_path=os.path.join(ann_dir, "instances_test.json"),
#         image_size=image_size,
#         class_id_to_idx=class_id_to_idx,
#     )
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
#
#     return train_loader, val_loader, test_loader, num_classes, idx_to_name


def build_datasets_and_dataloaders( resplit_root, image_size=(192, 192), batch_size=4, num_workers=8, persistent_workers=True, subset_train=None, subset_val=None, subset_test=None, subset_seed=42):
    """
    Builds dataset objects and dataloaders for train/val/test splits.

    Args:
        resplit_root (str): path to the resplit root
        image_size (tuple): (H, W) resize
        batch_size (int): batch size for all loaders
        num_workers (int): DataLoader workers
        subset_train (float|int|None): if float, fraction of train to use; if int, absolute count
        subset_val   (float|int|None): same for val
        subset_test  (float|int|None): same for test
        subset_seed (int): RNG seed for index sampling

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, idx_to_name)
    """
    ann_dir = os.path.join(resplit_root, "annotations")

    # Build a class map from train JSON (consistent with training)
    class_id_to_idx, idx_to_name, num_classes = build_class_map(
        os.path.join(ann_dir, "instances_train.json")
    )

    train_ds = SeaDrones_v2_Dataset(
        images_dir=os.path.join(resplit_root, "images", "train"),
        instances_json_path=os.path.join(ann_dir, "instances_train.json"),
        image_size=image_size,
        class_id_to_idx=class_id_to_idx,
    )
    val_ds = SeaDrones_v2_Dataset(
        images_dir=os.path.join(resplit_root, "images", "val"),
        instances_json_path=os.path.join(ann_dir, "instances_val.json"),
        image_size=image_size,
        class_id_to_idx=class_id_to_idx,
    )
    test_ds = SeaDrones_v2_Dataset(
        images_dir=os.path.join(resplit_root, "images", "test"),
        instances_json_path=os.path.join(ann_dir, "instances_test.json"),
        image_size=image_size,
        class_id_to_idx=class_id_to_idx,
    )

    # wrap with Subset if requested ---
    if subset_train is not None:
        idx_tr = _take_subset_indices(len(train_ds), subset_train, seed=subset_seed)
        train_ds = Subset(train_ds, idx_tr)
    if subset_val is not None:
        idx_va = _take_subset_indices(len(val_ds), subset_val, seed=subset_seed + 1)
        val_ds = Subset(val_ds, idx_va)
    if subset_test is not None:
        idx_te = _take_subset_indices(len(test_ds), subset_test, seed=subset_seed + 2)
        test_ds = Subset(test_ds, idx_te)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, num_classes, idx_to_name


def inspect_dataloaders(train_loader, val_loader, test_loader, log_file="results/data_summary.txt"):
    """Logs dataset sizes and inspects the first batch of the train dataloader."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    write_log("=== Dataloaders ===", filename=log_file)

    # sizes
    write_log(f"train: {len(train_loader.dataset)} samples, {len(train_loader)} batches", filename=log_file)
    write_log(f"val: {len(val_loader.dataset)} samples, {len(val_loader)} batches", filename=log_file)
    write_log(f"test: {len(test_loader.dataset)} samples, {len(test_loader)} batches", filename=log_file)
    write_log("", filename=log_file)

    # first batch from train
    for images, targets in train_loader:
        X = images[0]
        y = targets[0]

        write_log("Inspecting train dataloader", filename=log_file)
        write_log(f"  X shape: {tuple(X.shape)}", filename=log_file)

        # target shapes
        write_log(f"  y keys: {list(y.keys())}", filename=log_file)
        write_log(f"  y['boxes'] shape : {tuple(y['boxes'].shape)}", filename=log_file)
        write_log(f"  y['labels'] shape: {tuple(y['labels'].shape)}", filename=log_file)
        if 'image_id' in y:
            write_log(f"  y['image_id'] shape: {tuple(y['image_id'].shape)}", filename=log_file)

        # X preview (values). Keep it small to avoid huge dumps.
        # Show a tiny top-left patch (e.g., all 3 channels, first 2 rows, first 5 cols).
        write_log(f"  First example of X value (patch [:, :2, :5]): {X[:, :2, :5]}", filename=log_file)

        # min/max after shapes (separate line as requested)
        write_log(f"  X min/max: ({X.min().item():.3f}, {X.max().item():.3f})", filename=log_file)

        # y previews (make it explicit these are previews)
        write_log(
            f"  boxes count: {y['boxes'].shape[0]} | labels count: {y['labels'].shape[0]}",
            filename=log_file
        )
        write_log(f"  First example's boxes: {y['boxes']}", filename=log_file)
        write_log(f"  First example's labels: {y['labels']}", filename=log_file)

        # batch sizes
        write_log(f"  First batch X len: {len(images)}", filename=log_file)
        write_log(f"  First batch y len: {len(targets)}", filename=log_file)

        # dtypes (include all requested)
        dtype_msg = f"  Tensor dtypes -> X: {X.dtype} | y['boxes']: {y['boxes'].dtype} | y['labels']: {y['labels'].dtype}"
        if 'image_id' in y:
            dtype_msg += f" | y['image_id']: {y['image_id'].dtype}"
        write_log(dtype_msg, filename=log_file)

        write_log("", filename=log_file)
        break


# all together

# def prepare_dataset(data_dir, seed=42, out_dir_name="resplit"):
#     """
#     Preparing the data pipeline:
#       a) resplit (write JSONs + copy images, log distributions)
#       b) build dataloaders
#       c) inspect dataloaders (log)
#     Returns: (train_loader, val_loader, test_loader num_classes, idx_to_name, resplit_root)
#     """
#     resplit_root = create_resplit_matching_ratios(data_dir, seed=seed, out_dir_name=out_dir_name)
#     train_loader, val_loader, test_loader, num_classes, idx_to_name = build_datasets_and_dataloaders(resplit_root)
#     inspect_dataloaders(train_loader, val_loader, test_loader, log_file="results/data_summary.txt")
#
#     return train_loader, val_loader, test_loader, num_classes, idx_to_name, resplit_root


def prepare_dataset(data_dir, seed=42, out_dir_name="resplit", subset_train=None, subset_val=None, subset_test=None, subset_seed=42):
    """
    Preparing the data pipeline:
      a) resplit (write JSONs + copy images, log distributions)
      b) build dataloaders (optionally on a subset)
      c) inspect dataloaders (log)
    Returns: (train_loader, val_loader, test_loader, num_classes, idx_to_name, resplit_root)
    """
    resplit_root = create_resplit_matching_ratios(data_dir, seed=seed, out_dir_name=out_dir_name)
    train_loader, val_loader, test_loader, num_classes, idx_to_name = build_datasets_and_dataloaders(
        resplit_root,
        subset_train=subset_train,
        subset_val=subset_val,
        subset_test=subset_test,
        subset_seed=subset_seed
    )
    inspect_dataloaders(train_loader, val_loader, test_loader, log_file="results/data_summary.txt")
    return train_loader, val_loader, test_loader, num_classes, idx_to_name, resplit_root
