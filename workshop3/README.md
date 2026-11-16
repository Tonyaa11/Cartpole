# Workshop 3: Reinforcement Learning for Failure Avoidance in CartPole

## 1. Overview

This folder contains the Workshop 3 project of the Machine Learning course (Semester 2026-III, Universidad Distrital Francisco José de Caldas, Group 5). Workshops 1 and 2 *predicted* whether a CartPole episode was about to fail; here an agent *controls* the same system to avoid those failures. The project includes the MDP formulation, a custom Gymnasium environment validated against CartPole-v1, a Deep Q-Network implemented from scratch in PyTorch, a hyperparameter sweep (6 configurations x 3 seeds), five baseline policies (one of them is the Workshop 1 classifier used as a policy), test evaluation on held-out initial states, and logging through TensorBoard.

**Authors:** Miguel Angel Torres Ramirez, Cristian David Romero Gil, Omar Esteban Peña Sanchez

| MDP component | Definition |
|---|---|
| State | `(x, x_dot, theta, theta_dot)`, 4 continuous values |
| Actions | `0` = push left (-10 N), `1` = push right (+10 N) |
| Transition | cart-pole equations of motion, Euler step of 0.02 s (identical to CartPole-v1) |
| Reward | +1 per step survived + optional potential-based shaping from the Workshop 1 finding (distance to the failure limits) |
| Termination | failure if \|theta\| > 12 degrees or \|x\| > 2.4 m; truncation at 500 steps |

Deliverables required by the workshop and where they are:

| Required | Location |
|---|---|
| `src/env.py` | `workshop3/src/env.py` |
| `src/agent.py` | `workshop3/src/agent.py` |
| `src/baselines.py` | `workshop3/src/baselines.py` |
| `src/train.py` | `workshop3/src/train.py` |
| `src/evaluate.py` | `workshop3/src/evaluate.py` |
| `checkpoints/agent_best.pt` | `checkpoints/agent_best.pt` |
| `runs/workshop3/` | `runs/workshop3/` |
| Report | `workshop3/report/main.tex` -> `group5_workshop3.pdf` |

The Workshop 3 code lives in `workshop3/` because the repository root already contains the Workshop 1 files (for example `src/baselines.py`).

---

## 2. Requirements

### System

* Linux (Ubuntu 20.04+ recommended), macOS, Windows or Google Colab
* Python 3.10+ (developed with Python 3.13; Colab works out of the box)
* CPU only: the Q-network has about 17 thousand parameters, so a GPU brings no speed-up

### Dependencies

Install the required packages:

```bash
pip install torch
pip install "gymnasium>=1.0"
pip install tensorboard
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
```

---

Or use the requirements.txt

```bash
pip install -r workshop3/requirements.txt
```

---

## 3. Environment Setup (Linux)

Create and activate a virtual environment:

```bash
python3 -m venv rl_env
source rl_env/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install dependencies (see section above).

### Google Colab

Open `workshop3/notebooks/workshop3_colab.ipynb` in Colab (File -> Open notebook -> GitHub) and run all the cells. It clones the repository, runs every step below in order and downloads the results.

---

## 4. Project Structure

```text
.
├── src/                          # Workshop 1 (reused: dataset.py, features.py, baselines.py)
├── data/                         # Workshop 1 dataset and split (used by the W1 classifier)
├── checkpoints/
│   ├── agent_best.pt             # best DQN agent (required deliverable)
│   └── workshop3_sweep/          # best checkpoint of every configuration x seed
├── runs/
│   └── workshop3/                # TensorBoard logs, CSV logs, JSON results
└── workshop3/
    ├── configs/
    │   └── sweep.json            # the 6 configurations of the hyperparameter search
    ├── src/
    │   ├── env.py                # custom Gymnasium environment + validation
    │   ├── agent.py              # DQN agent (Q-network, replay buffer, target network)
    │   ├── baselines.py          # random, alternating, angle rule, PD rule, W1-guided
    │   ├── train.py              # training loop, validation, sweep and selection
    │   ├── evaluate.py           # test evaluation (nominal and noisy)
    │   ├── w1_bridge.py          # rebuilds the Workshop 1 classifier
    │   ├── common.py             # paths and seeds shared by all scripts
    │   └── make_report_assets.py # figures and LaTeX tables of the report
    ├── notebooks/
    │   └── workshop3_colab.ipynb # step-by-step Colab runner
    ├── report/
    │   ├── main.tex              # IEEE report (Overleaf)
    │   ├── figures/              # generated figures
    │   └── generated/            # generated numbers and table rows
    ├── requirements.txt
    └── README.md
```

---

## 5. Running Experiments

Run every command from the repository root.

### 5.1 Validating the Environment

```bash
python workshop3/src/env.py
```

This will:

* Check the Gymnasium API with `check_env`
* Compare the dynamics with Gymnasium's CartPole-v1 (same seeds and actions must give the same states)
* Run 100 random-policy episodes and print the mean return, the episode-length distribution and the action frequency
* Save the diagnostics in `runs/workshop3/env_validation.json`

---

### 5.2 Checking the Workshop 1 Classifier (optional)

```bash
python workshop3/src/w1_bridge.py
```

Rebuilds the Workshop 1 Gradient Boosting model (same split, features, scaler and threshold) and verifies that the fast feature code used inside the control loop matches `src/features.py`.

---

### 5.3 Training a Single Configuration

```bash
python workshop3/src/train.py --config reference --seed 42
```

---

### 5.4 Running the Hyperparameter Sweep

```bash
python workshop3/src/train.py --sweep
```

This will:

* Train the 6 configurations of `workshop3/configs/sweep.json` with seeds 42, 7 and 99 (300 episodes each)
* Evaluate the greedy agent every 10 episodes on 10 validation initial states and keep the best checkpoint of each run in `checkpoints/workshop3_sweep/`
* Select the best configuration on validation results only and copy its best run to `checkpoints/agent_best.pt`
* Write `runs/workshop3/sweep_results.json`

To redo only the selection from the saved logs: `python workshop3/src/train.py --select-only`.

---

### 5.5 Test Evaluation

```bash
python workshop3/src/evaluate.py
```

This will:

* Evaluate the agent and the five baselines on 100 held-out initial states, in the nominal environment and with actuator noise (2 N)
* Report return, success rate, failure causes, W1 danger rate and mean |theta|
* Write `runs/workshop3/test_results.json` and `runs/workshop3/trajectories.json`

---

### 5.6 Figures and LaTeX Tables

```bash
python workshop3/src/make_report_assets.py
```

Writes the figures to `workshop3/report/figures/` and every number of the report to `workshop3/report/generated/`. Upload the whole `workshop3/report/` folder to Overleaf and compile `main.tex`.

---

## 6. TensorBoard Visualization

To visualize training logs:

```bash
tensorboard --logdir runs/workshop3 --port 6006
```

Then open in your browser:

```text
http://localhost:6006
```

In Colab: `%load_ext tensorboard` and `%tensorboard --logdir runs/workshop3`.

Logs are organized as:

```text
runs/workshop3/
  ├── reference/
  │   ├── seed_42/   # TensorBoard events + episodes.csv + validation.csv + summary.json
  │   ├── seed_7/
  │   └── seed_99/
  ├── lr_1e-4/
  ...
```

Logged per episode: task return, shaped return, episode length, epsilon and TD loss; per validation: validation return, smoothed validation return and how the validation episodes ended.

---

## 7. Reproducibility

All experiments are reproducible due to:

* Fixed seeds: training runs 42, 7 and 99; validation initial states 10000-10009; test initial states 20000-20099 (three disjoint sets)
* Configurations defined in `workshop3/configs/sweep.json`
* A fixed training budget (300 episodes per run)
* Configuration selection with validation results only; the test set is used only by `evaluate.py`
* Every number and figure of the report generated from the logs by `make_report_assets.py`

To reproduce results:

1. Set up the environment
2. Run sections 5.1, 5.4, 5.5 and 5.6 in order
3. Visualize the runs in TensorBoard

Neural-network training can change with the library versions and the hardware, so the numbers of the report come from the runs whose logs and checkpoints are stored in `runs/workshop3/` and `checkpoints/`. Running `train.py --select-only`, `evaluate.py` and `make_report_assets.py` on those stored runs reproduces every number of the report; retraining produces new runs whose numbers can differ.

---

## 8. Logging Artifacts

The `runs/workshop3/` directory contains:

* TensorBoard events and CSV logs of the 18 training runs
* `sweep_results.json`: per-run summaries, per-configuration table and the selection
* `test_results.json`: test metrics of every policy in both test conditions
* `trajectories.json`: example test episodes of the selected agent
* `env_validation.json`: random-policy diagnostics

They can be inspected without retraining.

---

## 9. Notes

* The full sweep takes about 20 to 40 minutes on a CPU; the test evaluation about 3 minutes (the W1-guided policy is the slowest).
* DQN degrades after reaching a good policy in every configuration; the agent used is the best validation checkpoint of each run, not the last one.
* A simple PD rule (`theta + 0.5 * theta_dot`) already solves the nominal task: it is reported as a baseline on purpose.
