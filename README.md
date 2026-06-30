# 🚀 PSR Multitask Chaos Prediction

!\[Python](https://img.shields.io/badge/Python-3.10-blue)
!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
!\[Status](https://img.shields.io/badge/Status-Research-blue)
!\[License](https://img.shields.io/badge/License-MIT-green)

\---

## 🧠 Overview

This repository provides a **complete, research-grade Python implementation** of an attention-based multitask learning framework for **early detection of instability and chaos** in nonlinear dynamical systems.

The framework integrates:

* Phase-space reconstruction (Takens embedding)
* Multitask deep learning (classification + forecasting)
* Attention-based temporal modeling
* Early-warning decision analysis

\---

## 🔬 Key Features

✔ Full Lorenz system simulation with parameter sweep  
✔ Eigenvalue-based instability analysis (Hopf region)  
✔ Phase-space reconstruction from scalar signals  
✔ Three-regime classification framework  
✔ Multitask learning (classification + forecasting)  
✔ Attention-enhanced classification branch  
✔ Grouped cross-validation by simulation run  
✔ Comprehensive evaluation (ROC, PR, calibration, etc.)  
✔ Publication-ready figures and tables

\---

## ⚙️ Repository Structure

```
psr-multitask-chaos-prediction/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   ├── run\\\_experiment.py
│   └── run\\\_fast\\\_smoke\\\_test.py
└── src/
    └── psrchaos/
        ├── \\\_\\\_init\\\_\\\_.py
        └── full\\\_pipeline.py
```

\---

## 🔄 Workflow

```
Scalar Signal x(t)
        ↓
Phase-Space Reconstruction (PSR)
        ↓
Sliding Windows
        ↓
Shared Encoder (RNN)
       / \\\\
      /   \\\\
Classification   Forecasting
      ↓             ↓
Regime Label   Future Trajectory
      ↓             ↓
Early Warning + Evaluation
```

\---

## 🔧 Installation

```
conda create -n psr-chaos python=3.10 -y
conda activate psr-chaos
pip install -r requirements.txt
```

\---

## ▶️ Run

```
PYTHONPATH=src python scripts/run\\\_experiment.py
```

\---

## 📊 Outputs

Results saved in:

```
psr\\\_ssm\\\_outputs/
```

Includes plots, metrics, and evaluation files.

\---

## 📄 Citation

If you use this repository in your research, please cite:

> **Hamid Ismail, Ahmad Harb, and Marwan Bikdash.**  
> *Attention-Based Multitask Learning with Phase-Space Reconstruction for Early Detection of Instability and Chaos in the Lorenz System.*  
> **IEEE Access**, 2026.  
> **Paper:** https://ieeexplore.ieee.org/abstract/document/11570096

```bibtex
@article{Ismail2026IEEEAccess,
  author  = {Hamid Ismail and Ahmad Harb and Marwan Bikdash},
  title   = {Attention-Based Multitask Learning with Phase-Space Reconstruction for Early Detection of Instability and Chaos in the Lorenz System},
  journal = {IEEE Access},
  year    = {2026},
  url     = {https://ieeexplore.ieee.org/abstract/document/11570096},
  doi     = {10.1109/ACCESS.2026.3596339}
}
```

\---

## 📬 Contact

Hamid D. Ismail  
North Carolina A\&T State University

