# Spatial Graph Structure Is Essential
## An Ablation Study of Graph Neural Networks for Tumor Microenvironment Characterization in Breast Histopathology

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the official implementation and experimental code for the paper:

> **Spatial Graph Structure Is Essential: An Ablation Study of Graph Neural Networks for Tumor Microenvironment Characterization in Breast Histopathology**
>
> *Submitted to Laboratory Investigation (IF=5.0)*

### Abstract

We present a systematic two-level ablation framework using the BreCAHAD breast histopathology dataset (162 H&E images, 23,496 annotated cells, 17 cases). At the image level, we compare traditional morphological, topological, and deep features. At the node level, we construct cell spatial graphs via Delaunay triangulation and evaluate GATv2 against MLP baselines through Full Graph vs. No Edges vs. Random Edges ablation. Our key finding: spatial graph structure contributes a mean ΔAUC of +0.336 (Wilcoxon p=0.031, Cohen's d=12.0) over feature-only baselines.

### Repository Structure

```
├── code/                    # All analysis scripts (19 Python files, ~7,700 lines)
│   ├── 01_data_preprocessing.py
│   ├── 02_topognn_model.py
│   ├── 02b_topognn_v2.py
│   ├── 02c_v3_part1_features.py
│   ├── 02c_v3_part2_gnn.py
│   ├── 02d_final_partA.py
│   ├── 02d_final_partB.py
│   ├── 02e_v4_partA.py
│   ├── 02e_v4_partB.py
│   ├── 02e_v4_partC.py
│   ├── 02f_v5_compute.py
│   ├── 02f_v5_visualize.py
│   ├── 02f_v6_visualize_main.py
│   ├── 02f_v6_visualize_supp.py
│   ├── 02f_v7_fig1_delaunay.py
│   ├── 02f_v7_radius_sensitivity.py
│   ├── 02g_fold_level_analysis.py
│   ├── 02h_convert_tiff.py
│   ├── 03_contrastive_learning.py
│   └── 04_multitask_gat.py
├── data/                    # Result data files (CSV, JSON)
├── figures/                 # Main figures (Fig1-6, 300 DPI TIFF)
├── supplementary/           # Supplementary figures (S1-S6, 300 DPI TIFF)
├── manuscript.md            # Full manuscript
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Results

| Metric | Value |
|--------|-------|
| Graph Structure Contribution (ΔAUC) | +0.336 |
| Wilcoxon signed-rank test | p = 0.031 |
| Cohen's d effect size | 12.0 (large) |
| Mitosis Proximity AUC | 0.984 [0.979-0.989] |
| Apoptosis Proximity AUC | 0.944 [0.932-0.955] |
| Tubule Proximity AUC | 0.923 [0.914-0.930] |
| Model Parameters | 185K |
| Inference Time | <5 ms/image |

### Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.8
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- NumPy, SciPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- OpenCV (for image processing)
- torchvision (for ResNet features)

### Dataset

The BreCAHAD dataset is publicly available at:
https://github.com/alinot/BreCAHAD

### Citation

If you use this code in your research, please cite:

```bibtex
@article{author2026spatial,
  title={Spatial Graph Structure Is Essential: An Ablation Study of Graph Neural Networks for Tumor Microenvironment Characterization in Breast Histopathology},
  author={[Author Names]},
  journal={Laboratory Investigation},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX}
}
```

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Contact

[Author contact information]
