# Sea-SAR Detectors

This project implements and compares different object detection models (Custom CNN, YOLO, Faster R-CNN, Transformer) on the [SeaDronesSee Object Detection v2 (Compressed)](https://seadronessee.cs.uni-tuebingen.de/dataset) dataset. The goal is to evaluate performance on small-object maritime detection in 4K aerial imagery, with relevance to search-and-rescue and maritime surveillance.

## Contents

- `src/` - Core Python code (preprocessing, models, trainers, evaluation, single prediction)
- `main.py` - Main script to run the full pipeline
- `data` - Raw data, processed COCO-format splits, tiles
- `results` - Saved metrics, logs, figures and predictions
- `requirements.txt` - Dependencies
- `predict_commands.txt` - Sample commands for prediction
- `README.md` - Project documentation
- `report.pdf` - Final report with methods, experiments, and results

## Installation

Clone the repo and install dependencies:

```
# works on macOS and Windows
git clone https://github.com/GiX007/sea-sar-detectors.git
cd sea-sar-detectors
pip install -r requirements.txt
```

## Quickstart

Run the entire project (EDA, preprocessing, training and evaluation):

```
python main.py
```

## Models Implemented

- **YOLOv1(simple baseline)**
- **Faster R-CNN (two-stage CNN detector, MobileNetV3-Large FPN backbone)**
- **RetinaNet (one-stage CNN detector with focal loss, ResNet-50 backbone)**
- **YOLOv8 (real-time CNN detector)**
- **DETR (transformer-based detector, ResNet-50 backbone)**

## Results

We evaluate YOLOv1, Faster R-CNN, RetinaNet, YOLOv8, and DETR on a re-split version of the SeaDronesSee dataset.  
Metrics are reported on the new annotated test split, using IoU@0.5, Precision@0.5, Recall@0.5, mAP@0.5, training time, inference time, and model size.  
All evaluation metrics, training logs, loss/IoU curves, and annotated predictions are saved under `results/`.

### Evaluation Results

| Model        |   IoU@0.5 |   Precision@0.5 |   Recall@0.5 |   mAP@0.5 |   Training Time (s) |   Average Inference Time (s) |   #Params |
|--------------|-----------|-----------------|--------------|-----------|---------------------|------------------------------|-----------|
| YOLOv1       |    0.0023 |          0.0004 |       0.0011 |  nan      |           1281.4873 |                       0.0074 |  24734367 |
| Faster R-CNN |    0.6160 |          0.2467 |       0.5431 |    0.3512 |           2550.3647 |                       0.0600 |  16003177 |
| RetinaNet    |    0.6765 |          0.1929 |       0.7579 |    0.3433 |          11663.8749 |                       0.3227 |   8776017 |
| YOLOv8       |    0.5063 |          0.5278 |       0.3078 |    0.4127 |          16962.7888 |                       0.0353 |   2685343 |
| DETR         |    0.2422 |          0.0172 |       0.1130 |    0.0073 |           1476.7855 |                       0.0284 |  18047754 |

**Summary:**  
RetinaNet delivered the strongest detection performance overall, with the highest IoU (0.68) and recall (0.76), though at the cost of very long training time and relatively slow inference. YOLOv8 offered the best trade-off: highest precision (0.53), competitive mAP (0.41), and the fastest inference (0.035 s) despite being the smallest model (2.7M params). Faster R-CNN achieved balanced performance (IoU 0.62, mAP 0.35) with moderate size and speed. DETR underperformed on this dataset (mAP 0.007) **proving that transformer-based detectors require larger datasets and more compute to outperform CNN-based approaches in practice**. YOLOv1 failed to converge.  

### Baseline configs used

- **YOLOv1**: simplified architecture (B=2), trained from scratch on SeaDronesSee  
- **Faster R-CNN**: MobileNetV3-Large FPN backbone with COCO-pretrained weights 
- **RetinaNet**: ResNet-50 backbone with ImageNet-pretrained weights, focal loss for class imbalance 
- **YOLOv8-S**: COCO-pretrained weights, fine-tuned on SeaDronesSee  
- **DETR**: ResNet-50 backbone with COCO-pretrained weights 

**Hardware:**  
All experiments were run with PyTorch 2.5.1+cu121 on a single NVIDIA GeForce GTX 1050 Ti (4 GB VRAM, CUDA 12.1, cuDNN 9.1).  

## Notes

- The dataset used is **SeaDronesSee Object Detection v2 (Compressed)** (≈9 GB, JPEG images).  
- The official test set annotations are withheld by the dataset provider, so we perform a local train/validation split and report results only on the accessible validation data.  
- All models are compared using consistent preprocessing and standard hyperparameters (no additional tuning).  
- Reported results may vary depending on hardware, random seed, and exact configuration.  

## Prediction Example

Run inference on a single image with any trained model using `src/predict_single.py`.  
Try different models and score thresholds to **compare results on the same image**.
Predictions are printed in the terminal, and annotated images are saved to `results/test_images_preds/`.

```
# Faster R-CNN
python -m src.predict_single `
  --model_type fasterrcnn `
  --model_path results/models/fasterrcnn.pth `
  --image_path data/test_images/209.jpg `
  --class_map results/models/class_map.json `
  --score_thr 0.6
```

```
# RetinaNet
python -m src.predict_single `
  --model_type retinanet `
  --model_path results/models/retinanet.pth `
  --image_path data/test_images/209.jpg `
  --class_map results/models/class_map.json `
  --score_thr 0.6
```

```
# YOLOv8
python -m src.predict_single `
  --model_type yolov8 `
  --model_path results/models/yolov8.pt `
  --image_path data/test_images/209.jpg `
  --class_map results/models/class_map.json `
  --score_thr 0.6
```

## Contributing

Contributions and feedback are welcome! Feel free to open issues or submit pull requests.
