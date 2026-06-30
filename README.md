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
       /           \
      /             \
Classification    Forecasting
      ↓               ↓
Regime Label    Future Trajectory
      ↓               ↓
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

## Additional Experiments

The repository also includes additional experiments designed to address generalization, baselines, loss-weight selection, robustness, and computational cost.

Run the full additional experiment suite:

```bash
PYTHONPATH=src python scripts/run_additional_experiments.py
```

For a quick verification run:

```bash
PYTHONPATH=src python scripts/run_additional_experiments.py --fast
```

The outputs are saved to:

```text
psr_ssm_outputs/additional_experiments/
├── independent_parameter_region_test.csv
├── baseline_model_comparison.csv
├── loss_weight_sensitivity.csv
├── multitask_gradient_cosine_similarity.csv
├── noise_robustness.csv
├── latent_space_pca.csv
├── computational_complexity.csv
├── additional_systems_summary.csv
└── *.png
```

These files support the revision by adding:

- independent held-out parameter-region testing,
- baseline comparisons with LSTM, CNN-LSTM, Transformer, and ESN,
- loss-weight sensitivity analysis,
- multitask gradient-alignment diagnostics,
- noise robustness evaluation,
- latent-space visualization of class separability,
- computational complexity and inference-time reporting,
- additional nonlinear-system simulation support for Rossler, Duffing, and Mackey-Glass systems.


## 📄 Citation
If you use this repository in your research, please cite:
> **Hamid Ismail, Ahmad Harb, and Marwan Bikdash.**  
> *Attention-Based Multitask Learning with Phase-Space Reconstruction for Early Detection of Instability and Chaos in the Lorenz System.*  
> **IEEE Access**, 2026.  
> **Paper:** https://ieeexplore.ieee.org/abstract/document/11570096

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
