<<<<<<< HEAD
# 🚀 PSR Multitask Chaos Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Research-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Overview

This repository provides a **complete, research-grade Python implementation** of an attention-based multitask learning framework for **early detection of instability and chaos** in nonlinear dynamical systems.

The framework integrates:
- Phase-space reconstruction (Takens embedding)
- Multitask deep learning (classification + forecasting)
- Attention-based temporal modeling
- Early-warning decision analysis

---

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

---

## ⚙️ Repository Structure

```
psr-multitask-chaos-prediction/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   ├── run_experiment.py
│   └── run_fast_smoke_test.py
└── src/
    └── psrchaos/
        ├── __init__.py
        └── full_pipeline.py
```

---

## 🔄 Workflow

```
Scalar Signal x(t)
        ↓
Phase-Space Reconstruction (PSR)
        ↓
Sliding Windows
        ↓
Shared Encoder (RNN)
       /          \
      /            \
Classification   Forecasting
      ↓             ↓
Regime Label   Future Trajectory
      ↓             ↓
Early Warning + Evaluation
```

---

## 🔧 Installation

```
conda create -n psr-chaos python=3.10 -y
conda activate psr-chaos
pip install -r requirements.txt
```

---

## ▶️ Run

```
PYTHONPATH=src python scripts/run_experiment.py
```

---

## 📊 Outputs

Results saved in:

```
psr_ssm_outputs/
```

Includes plots, metrics, and evaluation files.

---

## 📄 Citation

```
H. Ismail, A. Harb, and M. Bikdash,
“Attention-Based Multitask Learning with Phase-Space Reconstruction for Early Detection of Instability and Chaos in the Lorenz System,” 2026.
```

```bibtex
@article{ismail2026psrchaos,
  title={Attention-Based Multitask Learning with Phase-Space Reconstruction for Early Detection of Instability and Chaos in the Lorenz System},
  author={Ismail, Hamid and Harb, Ahmad and Bikdash, Marwan},
  year={2026}
}
```

---

## 📬 Contact

Hamid Ismail  
North Carolina A&T State University
=======
# psr-multitask-chaos-prediction
A hybrid deep learning framework for early detection of instability and chaos in nonlinear dynamical systems using phase-space reconstruction, multitask learning, and attention-based classification. Includes simulation of the Lorenz system, dataset generation, forecasting, and early-warning evaluation.
>>>>>>> 8808b3c77137c1b523035b9b8d4fff5dbe92dbec
