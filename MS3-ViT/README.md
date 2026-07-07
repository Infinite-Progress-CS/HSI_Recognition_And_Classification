# MS3-ViT: Multi-Scale Spectral-Spatial Vision Transformer

**Open-Set Semi-Supervised Hyperspectral Image Classification**

## Architecture Overview

```
HSI Cube (B, C, S, S)
    │
    ▼
[1] SpectralGroupingLayer (SGL) — 物理光谱分组
    将C个波段按电磁波谱分组，组内自注意力+组间交叉注意力
    │
    ▼
[2] MultiScalePatchEmbedding — 4支路并行
    3×3, 5×5, 7×7, 9×9 不同感受野
    │
    ▼
[3] MultiScaleViTEncoder — 每尺度2层轻量Transformer
    │
    ▼
[4] CrossScaleGatedFusion (CSGF) — MSPF-ViT门控融合
    可学习权重 + 跨尺度注意力增强
    │
    ▼
[5] ClassAwareOpenSetHead — 类别感知开集分类
    原型对比 + 类别自适应阈值(CADT) + 已知类采样器
    │
    ▼
输出: {0..K-1} ∪ {unknown}
```

## File Structure

```
MS3-ViT/
├── config.py               # 所有超参数配置
├── data_loader.py          # HSI数据加载和增强
├── losses.py               # 损失函数(L_sup + L_cons + L_pco)
├── train.py                # 训练主循环
├── utils.py                # 评估指标和可视化
├── model/
│   ├── __init__.py
│   ├── spectral_grouping.py    # 模块1: SGL
│   ├── multi_scale_patch.py    # 模块2: 多尺度嵌入
│   ├── per_scale_vit.py        # 模块3: 每尺度ViT
│   ├── csgf.py                 # 模块4: 跨尺度门控融合
│   ├── open_set_head.py        # 模块5: 开集分类头
│   └── ms3_vit.py              # 完整模型集成
└── README.md
```

## Quick Start

```bash
# Train on Indian Pines
python train.py --dataset IndianPines --gpu 0

# Train on Pavia University
python train.py --dataset PaviaU --gpu 0

# Train on Salinas
python train.py --dataset Salinas --gpu 0
```

## Key Innovations

1. **Spectral Grouping Layer (SGL)**: Physically-motivated band grouping
   based on electromagnetic spectrum (VIS/VNIR/SWIR1/SWIR2)

2. **Multi-Scale Parallel ViT**: 4 parallel branches with different
   spatial receptive fields (3x3, 5x5, 7x7, 9x9), inheriting the
   MSPF-ViT multi-scale fusion mechanism

3. **Cross-Scale Gated Fusion (CSGF)**: Learnable gated weights with
   optional cross-scale attention — direct evolution of MSPF-ViT fusion

4. **Class-Adaptive Deviation Threshold (CADT)**:
   Per-class adaptive kappa instead of fixed kappa=2 (CACL)

## Datasets

Place `.mat` files in `D:\本科生科研\HSI分类\HSI数据集\`:

- `Indian_pines_corrected.mat` + `Indian_pines_gt.mat`
- `PaviaU.mat` + `PaviaU_gt.mat`
- `Salinas_corrected.mat` + `Salinas_gt.mat`

## Requirements

- Python 3.8+
- PyTorch 1.11+
- scipy, numpy, scikit-learn, matplotlib

## References

- **MSPF-ViT**: Han Gao, "Multi-Scale Parallel Feature Fusion ViT
  for Image Classification" (EI Conference)
- **CACL**: Sun et al., "Class-Aware Consistency Learning for Open-Set
  Semi-Supervised HSI Classification" (IEEE TGRS, 2025)
- **SSMLP-RPL**: Sun et al., "Spectral-Spatial MLP-Like Network with
  Reciprocal Points Learning" (IEEE TGRS, 2023)
- **DSCA-Net**: Lu et al., "Dual-Stream Class-Adaptive Network for
  Semi-Supervised HSI Classification" (IEEE TGRS, 2024)
