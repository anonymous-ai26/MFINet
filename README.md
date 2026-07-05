# MFINet: Multi-view Fusion and 2D-3D Interaction Enhancement for Real-Time LiDAR Semantic Segmentation

MFINet is a real-time LiDAR semantic segmentation network based on multi-view fusion and 2D-3D interaction enhancement. It integrates 3D Point View (3D-PV), 2D Bird's Eye View (2D-BEV), and 2D Range View (2D-RV) to make full use of complementary 3D and 2D representations.


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
