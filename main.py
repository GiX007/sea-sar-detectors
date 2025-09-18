# src/main.py
#
# Main script for running Sea-SAR detection results.
# Runs eda, preprocessing, model training, evaluation, and saves results.
#
import os, torch
import pandas as pd
from src.utils import write_log, log_device_info
from src.eda import run_eda
from src.preprocess import prepare_dataset

from src.models.yolo_v1 import run_yolo_v1
from src.models.faster_rcnn import run_faster_rcnn
from src.models.retinanet import run_retinanet
from src.models.yolo_v8 import run_yolo_v8
from src.models.detr import run_detr


# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main():
    """Main driver function for the detection results pipeline."""

    # container for all results
    all_results = []

    # 1. EDAI hve
    print("\n>>> Running EDA ...")
    run_eda(DATA_DIR)

    # 2. Resplitting and building dataloaders
    print("\n>>> Resplitting and building dataloaders ...")
    # train_loader, val_loader, test_loader, num_classes = prepare_dataset(DATA_DIR)
    train_loader, val_loader, test_loader, num_classes, idx_to_name, resplit_root = prepare_dataset(
        data_dir="data", seed=42, out_dir_name="resplit",
        subset_train=0.10, # 0.15, # ~ 990 train
        subset_val=0.1, #0.20, # ~ 228 val
        subset_test=0.25 # 0.33 # ~ 912 test
    )

    # set the device, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device)

    # 3. Run YOLOv1 baseline
    print("\n>>> Training and Evaluating YOLOv1 ...")
    yolo_v1_results = run_yolo_v1(train_loader, val_loader, test_loader, num_classes, device)
    all_results.extend(yolo_v1_results)

    # 4. Run Faster R-CNN
    print("\n>>> Training and Evaluating Faster R-CNN ...")
    faster_rcnn_results = run_faster_rcnn(train_loader, val_loader, test_loader, num_classes, device)
    all_results.extend(faster_rcnn_results)

    # 5. Run RetinaNet
    print("\n>>> Training and Evaluating RetinaNet ...")
    retinanet_results = run_retinanet(train_loader, val_loader, test_loader, num_classes, device)
    all_results.extend(retinanet_results)

    # 6. Run YOLOv8
    print("\n>>> Training and Evaluating YOLOv8 ...")
    yolo_names = idx_to_name[1:] # YOLOv8 wants names 0..K-1 (no background). Our idx_to_name[0] = "__background__".
    yolo_v8_results = run_yolo_v8(train_loader, val_loader, test_loader, num_classes, device,
                                  resplit_root=resplit_root, class_names=yolo_names)
    all_results.extend(yolo_v8_results)

    # 7. Run DETR
    print("\n>>> Training and Evaluating DETR ...")
    detr_results = run_detr(train_loader, val_loader, test_loader, num_classes, device)
    all_results.extend(detr_results)

    # Store all results into a df and log
    columns = ["Model", "IoU@0.5", "Precision@0.5", "Recall@0.5", "mAP@0.5", "Training Time (s)", "Average Inference Time (s)", "#Params"]
    df = pd.DataFrame(all_results).reindex(columns=columns)

    table_str = df.to_markdown(index=False, tablefmt="github", floatfmt=".4f")
    write_log("==== Evaluation Results ====\n", filename="results/evaluation_results.txt")
    write_log(table_str + "\n", filename="results/evaluation_results.txt")
    print("\nAll evaluation results have been saved to: ./results/evaluation_results.txt\n")


if __name__ == "__main__":
    main()
