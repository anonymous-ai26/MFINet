# MFINet: Multi-view Fusion and 2D-3D Interaction Enhancement for Real-Time LiDAR Semantic Segmentation

MFINet is a real-time LiDAR semantic segmentation network based on multi-view fusion and 2D-3D interaction enhancement. It integrates 3D Point View (3D-PV), 2D Bird's Eye View (2D-BEV), and 2D Range View (2D-RV) to make full use of complementary 3D and 2D representations.

## Method Overview

MFINet contains four main components:

- **3D Point Feature Projector (3DPFP)**: projects point-wise 3D features into BEV and RV pseudo-images through voxel/grid pooling.
- **Feature Enhancement (FE)**: samples BEV/RV 2D features back to points to enhance point-wise geometric and semantic representation.
- **2D-3D Fusion Head (FH)**: aggregates point features from 3D-PV, 2D-BEV, and 2D-RV for final point-wise prediction.
- **Multi-Scale Dilated Attention (MSDA)**: enhances BEV/RV feature discrimination with local sliding-window dilated attention.

## Project Structure

```text
config/mfinet.py        MFINet training configuration
models/mfinet.py        Main MFINet model
networks/bird_view.py   BEV branch with MSDA and attention decoder
networks/range_view.py  RV branch with MSDA and attention decoder
networks/backbone.py    Basic blocks, point MLPs, fusion layers, and sampling layers
datasets/               SemanticKITTI dataset loader and preprocessing utilities
utils/                  Losses, metrics, logger, and optimizer/scheduler builder
deep_point/             CUDA extension for point-to-grid voxel pooling
```

## Environment Setup

Install the required Python packages and compile the CUDA extension in `deep_point` before training.

```bash
cd deep_point
python setup.py install
```

## Prepare Data

Download SemanticKITTI from the official website and set `SeqDir` in `config/mfinet.py` to your dataset path.

If copy-paste augmentation is enabled, prepare the object bank and set `ObjBackDir` in `config/mfinet.py`.

## Training

```bash
python train.py --config config/mfinet.py
```

## Evaluation

```bash
python evaluate.py --config config/mfinet.py --start_epoch 0 --end_epoch 49
```

## Find Best Epoch

```bash
python find_best_metric.py --name mfinet
```
