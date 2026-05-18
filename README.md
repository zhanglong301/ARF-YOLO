# ARF-YOLO: Attention-Guided Adaptive Resolution-Aware Feature Learning for UAV Remote Sensing Object Detection

## Introduction

ARF-YOLO is a novel UAV detection framework built upon YOLOv11 with three synergistic innovations:

- **AGRH (Attention-Guided Resolution Heads)**: Dual-resolution detection heads with Multi-Perspective Feature Attention (MPFA) for simultaneous exploitation of semantic and spatial features.
- **AMFF (Adaptive Multi-Level Feature Fusion)**: Content-adaptive dynamic kernel generation (FAUS), structure-guided feature refinement (FRS), and learned cross-level weighting (AFFS).
- **FSRA (Fast Scale Resource Assigner)**: Dynamic allocation of representation capacity along channels, spatial patches, and scale levels via three lightweight parallel assigners.
- **ARF-SALoss**: Instance-level inverse-scale weighting that amplifies supervisory signal for small objects.

## Performance

| Dataset | mAP@0.5 | mAP@0.5:0.95 | Params | FLOPs | FPS |
|---------|---------|---------------|--------|-------|-----|
| VisDrone2019 | 48.5% | 29.2% | 22.3M | 74.5G | 101 |
| UAVDT | 63.7% | 37.1% | 22.3M | 74.5G | 101 |

## Installation

```bash
# Clone the repository
git clone https://github.com/zhanglong301/ARF-YOLO.git
cd ARF-YOLO

# Create conda environment
conda create -n arf-yolo python=3.10 -y
conda activate arf-yolo

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### VisDrone2019
```bash
# Download VisDrone2019-DET dataset
# Place it in the following structure:
# datasets/
#   visdrone/
#     images/
#       train/
#       val/
#       test/
#     labels/
#       train/
#       val/
#       test/
```

### UAVDT
```bash
# Download UAVDT dataset
# Place it in the following structure:
# datasets/
#   uavdt/
#     images/
#       train/
#       val/
#     labels/
#       train/
#       val/
```

## Training

```bash
# Train on VisDrone2019
python train.py --cfg configs/arf_yolo.yaml --data configs/visdrone.yaml --epochs 200 --batch-size 16 --img-size 640

# Train on UAVDT
python train.py --cfg configs/arf_yolo.yaml --data configs/uavdt.yaml --epochs 200 --batch-size 16 --img-size 640
```

## Evaluation

```bash
# Evaluate on VisDrone2019
python val.py --weights runs/train/arf_yolo_visdrone/weights/best.pt --data configs/visdrone.yaml --img-size 640

# Evaluate on UAVDT
python val.py --weights runs/train/arf_yolo_uavdt/weights/best.pt --data configs/uavdt.yaml --img-size 640
```

## Inference

```bash
# Run inference on images
python detect.py --weights runs/train/arf_yolo_visdrone/weights/best.pt --source path/to/images --img-size 640
```

## Model Weights

| Model | Dataset | mAP@0.5 | Download |
|-------|---------|---------|----------|
| ARF-YOLO | VisDrone2019 | 48.5% | [weights](https://github.com/zhanglong301/ARF-YOLO/releases) |
| ARF-YOLO | UAVDT | 63.7% | [weights](https://github.com/zhanglong301/ARF-YOLO/releases) |

## Citation

```bibtex
@article{arfyolo2025,
  title={ARF-YOLO: Attention-Guided Adaptive Resolution-Aware Feature Learning for UAV Remote Sensing Object Detection},
  author={},
  journal={Journal of King Saud University - Computer and Information Sciences},
  year={2025}
}
```

## License

This project is released under the [GPL-3.0 License](LICENSE).

## Acknowledgements

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset)
- [UAVDT](https://sites.google.com/view/grli-uavdt)
