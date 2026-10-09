# CartPole Failure Prediction - Machine Learning, Workshop 1

**Course:** Machine Learning, Semester 2026-III - Universidad Distrital Francisco José de Caldas
**Workshop 1:** Data Preparation & Supervised Learning Baselines
**Group:** `<N>` - **Authors:** `<names>`

## 1. Problem

Binary classification on CartPole trajectories. Given the state of the system
`(x, x_dot, theta, theta_dot)` and the `action` taken at time `t`, predict whether
the episode is about to fail: `label = 1` in the **last 5 steps** of every episode
that ends in failure, `label = 0` otherwise.

## 2. Dataset

| Property | Value |
|---|---|
| File | `data/raw/cartpole_dataset.csv` |
| Size | ~2 MB (1.99 MB) |
| Rows / episodes | 35,874 / 400 |
| Columns | `episode, t, x, x_dot, theta, theta_dot, action, label` |
| Class balance | 94.51 % class 0 / 5.49 % class 1 (imbalance 17.2 : 1) |
| SHA256 | `ec067d5dcc498f2b5c5486d24f290283431ef3a0db0a4952cfde84f07e4f53f8` |

**Provenance / license:** `<TODO: describe how the dataset was generated (environment, policy, seed, script) or its source and license>`.

### Download / setup from scratch

```bash
git clone <repo-url> && cd cartpole-ml
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Place cartpole_dataset.csv in data/raw/ (already included in this repo)
python dataset_info.py                                  # verifies the SHA256 and prints a summary
```

## 3. Repository structure

```
cartpole-ml/
|-- dataset_info.py        # dataset scan: integrity, counts, ranges
|-- requirements.txt
|-- data/
|   |-- raw/               # cartpole_dataset.csv
|   |-- processed/
|   `-- splits.json        # episode ids of train / val / test (never re-split!)
|-- src/
|   |-- dataset.py         # loading, episode split, scaling, augmentation
|   |-- features.py        # feature engineering (past-only)
|   `-- baselines.py       # Logistic Regression, Random Forest, Gradient Boosting
|-- notebooks/eda.ipynb    # exploratory data analysis
|-- checkpoints/           # model weights (future workshops)
|-- runs/                  # results of the baselines (csv / json)
`-- report/main.tex        # IEEE-style report (Overleaf)
```

## 4. Reproduce all results

```bash
python src/dataset.py            # creates data/splits.json (seed 42)
python src/baselines.py          # main results        -> runs/baseline_results_base.csv
python src/baselines.py --raw-only   # ablation: raw features only
python src/baselines.py --augment    # ablation: noise augmentation (train only)
jupyter notebook notebooks/eda.ipynb # EDA
```

All scripts fix the seed (42) for `random` and `numpy`, and models receive the same seed.

## 5. Methodology summary

* **Split:** 70 / 15 / 15 **by episode** (279 / 61 / 60), stratified by episode-length quartile.
  Splitting by row would leak near-identical consecutive states between partitions.
* **Features (16):** raw state + action, distance to the failure limits
  (|theta| < 0.2095 rad, |x| < 2.4 m), first differences, 3-step rolling means and
  time-to-limit estimates. Only current/past information is used; `t` is excluded.
* **Scaling:** z-score fitted on the training set only.
* **Augmentation (optional, train only):** Gaussian sensor noise (2 % of each std).
* **Imbalance:** class weights (Logistic Regression, Random Forest) / sample weights (Gradient Boosting).
* **Threshold:** chosen on validation (max F1); test is never used for tuning.
* **Metrics:** precision, recall, F1, ROC-AUC, PR-AUC, accuracy and confusion matrix.
  The most important ones are **recall** (a missed failure is the costly error) and **PR-AUC**
  (robust to the 17:1 imbalance); accuracy alone is misleading.

## 6. Results (test set, threshold chosen on validation)

See `runs/baseline_results_*.csv` and the report in `report/main.tex`.
