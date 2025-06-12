# 🧪 Koopmanism and Response Theory

This code is for reproducing the paper "Bridging the Gap between Koopmanism and Response Theory: Using Natural Variability to Predict Forced Response" to appear in SIAM Journal on Applied Dynamical Systems. 

It performs Extended Dynamic Mode Decomposition to estimate the spectral properties of the Koopman/Kolmogorov operator, from which the response to perturbations of chaotic systems is estimated. 

## 🚀 Features

- Python project managed with [Poetry](https://python-poetry.org/)
- Code quality checks with [`pre-commit`](https://pre-commit.com/)
- Reproducible environments via `poetry.lock`
- Easy setup with `set_up.sh`

---

## 📦 Installation

> **Prerequisite**: You must have [Poetry installed](https://python-poetry.org/docs/#installation).  
> (Recommended via `pipx`: `pipx install poetry`)

Clone the repository and run the setup script:

```bash
git clone git@github.com:niccozagli/KoopmanismResponse.git
cd KoopmanismResponse
bash set_up.sh
```
---
##  How to use 

In the notebooks folder you can find the two notebooks to reproduce the results for the chaotic systems investigated in the paper. In the scripts folder you can find the shell scripts to run the direct numerical response experiments.

The numerical results on the double-well system can be found in the "double well" folder.