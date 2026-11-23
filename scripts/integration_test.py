"""
integration_test.py  —  End-to-end pipeline test for all four workshops.

Each workshop test loads real data/checkpoints and runs the pipeline at minimal
scale (a few episodes / one partition) to catch import errors, shape mismatches
and broken file paths before submitting. A first, fast "repository" check runs
for every call:

* every ``import`` written in ``src/`` and ``scripts/`` resolves from the ``src``
  root (this catches stale module names after moving files, even inside
  functions that the other tests never call);
* every sweep checkpoint listed in ``runs/workshop3/sweep_results.json`` exists;
* the stored Workshop 3 test results contain the W2 policy and the W2 danger
  rate that the Workshop 3 report quotes;
* the installed scikit-learn is the pinned version that reproduces the reports
  (a WARN, not a failure: the pipeline still runs, but tree-model numbers can
  change slightly).

Usage
-----
    python scripts/integration_test.py           # all workshops
    python scripts/integration_test.py --w 1 2   # specific workshops (repository checks always run)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

PINNED_SKLEARN = "1.6.1"   # must match requirements.txt

PASS = "  PASS"
FAIL = "  FAIL"
SKIP = "  SKIP"
WARN = "  WARN"


def _header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _check(label: str, cond: bool, detail: str = "") -> bool:
    """Print PASS / FAIL for one check and return the condition.

    Args:
        label (str): What is being checked.
        cond (bool): Result of the check.
        detail (str): Optional extra information.

    Returns:
        bool: ``cond`` (so checks can be chained with ``ok &= ...``).
    """
    icon = PASS if cond else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{icon}  {label}{suffix}")
    return cond


# ---------------------------------------------------------------------------
# Repository consistency (always run, a few seconds)
# ---------------------------------------------------------------------------

def unresolved_imports() -> list[str]:
    """Find absolute imports in ``src/`` and ``scripts/`` that do not resolve.

    Every script inserts ``src/`` into ``sys.path`` and imports the workshop
    packages as ``w1.*`` ... ``w4.*``. A bare ``import dataset`` or ``from
    sequences import ...`` only works by accident (when the script happens to
    run from that folder), so it is reported. Imports inside functions are
    checked too, because the AST is scanned, not executed.

    Returns:
        list[str]: One "file:line: module" entry per unresolved import.
    """
    bad = []
    files = sorted(list(SRC.rglob("*.py")) + list((ROOT / "scripts").glob("*.py")))
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                try:
                    found = importlib.util.find_spec(name) is not None
                except (ModuleNotFoundError, ValueError):
                    found = False
                if not found:
                    bad.append(f"{f.relative_to(ROOT)}:{node.lineno}: {name}")
    return bad


def scripts_without_src_path() -> list[str]:
    """Find modules that import the workshop packages but never add ``src/`` to ``sys.path``.

    When a file is run as a script (``python src/w3/env.py``) Python only puts its
    own folder on the path, so ``from w3.common import ...`` works only if the file
    first inserts ``src/`` (the convention of every script of the repository).

    Returns:
        list[str]: Files that import ``w1``-``w4`` without that line.
    """
    marker = "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))"
    bad = []
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        uses_pkg = any(
            (isinstance(n, ast.ImportFrom) and n.level == 0 and (n.module or "").split(".")[0] in {"w1", "w2", "w3", "w4"})
            or (isinstance(n, ast.Import) and any(a.name.split(".")[0] in {"w1", "w2", "w3", "w4"} for a in n.names))
            for n in ast.walk(tree))
        if uses_pkg and marker not in f.read_text(encoding="utf-8"):
            bad.append(str(f.relative_to(ROOT)))
    return bad


def test_repo() -> bool:
    """Static checks that need no training: imports, paths, stored results."""
    _header("Repository — imports, paths, stored results")
    ok = True

    # Run with only src/ (and the installed packages) on the path, as the scripts do
    saved_path = list(sys.path)
    sys.path[:] = [str(SRC)] + [p for p in saved_path
                                if p and Path(p).resolve() not in (ROOT, ROOT / "scripts")
                                and not Path(p).resolve().is_relative_to(SRC)]
    try:
        bad = unresolved_imports()
    finally:
        sys.path[:] = saved_path
    ok &= _check("every import resolves from src/", not bad,
                 "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))
    no_path = scripts_without_src_path()
    ok &= _check("modules importing w1-w4 add src/ to sys.path (run as scripts)", not no_path,
                 ", ".join(no_path))

    sweep_file = ROOT / "runs" / "workshop3" / "sweep_results.json"
    if sweep_file.exists():
        from w3.common import SWEEP_CKPT_DIR
        runs = json.loads(sweep_file.read_text())["runs"]
        missing = [r["checkpoint"] for r in runs
                   if not (SWEEP_CKPT_DIR / Path(r["checkpoint"]).name).exists()]
        ok &= _check(f"W3 sweep checkpoints present ({len(runs)} runs)", not missing,
                     ", ".join(missing[:3]))

    test_file = ROOT / "runs" / "workshop3" / "test_results.json"
    if test_file.exists():
        test = json.loads(test_file.read_text())
        pols = test["conditions"]["nominal"]["policies"]
        names = [p["policy"] for p in pols]
        ok &= _check("W3 test results include the W2 policy and W2 danger rate",
                     "W2 LSTM (look-ahead)" in names and all("w2_danger_rate" in p for p in pols)
                     and "w2_threshold" in test)

    try:
        import sklearn
        if sklearn.__version__ != PINNED_SKLEARN:
            print(f"{WARN}  scikit-learn {sklearn.__version__} installed, reports use {PINNED_SKLEARN}: "
                  "Random Forest / Gradient Boosting numbers may differ slightly "
                  f"(pip install scikit-learn=={PINNED_SKLEARN})")
        else:
            _check(f"scikit-learn {sklearn.__version__} (pinned version)", True)
    except ImportError:
        ok &= _check("scikit-learn installed", False)
    return ok


# ---------------------------------------------------------------------------
# Workshop 1
# ---------------------------------------------------------------------------

def test_w1() -> bool:
    """Data loading, features and the three baselines (threshold from validation)."""
    _header("Workshop 1 — Data + Baselines")
    ok = True

    # 1. Data loading
    try:
        from w1.dataset import load_raw, load_splits
        df = load_raw()
        splits = load_splits()
        ok &= _check("load_raw()", df.shape[0] > 1000, f"{df.shape[0]} rows")
        ok &= _check("load_splits()", set(splits.keys()) == {"train", "val", "test"})
    except Exception:
        print(FAIL + "  load_raw / load_splits")
        traceback.print_exc()
        return False

    # 2. Feature engineering
    try:
        from w1.features import build_features
        df_feat = build_features(df)
        ok &= _check("build_features()", df_feat.shape[1] > df.shape[1],
                     f"{df_feat.shape[1]} columns")
    except Exception:
        print(FAIL + "  build_features")
        traceback.print_exc()
        ok = False

    # 3. Train all three baselines end-to-end (threshold chosen on VALIDATION)
    try:
        from w1.baselines import prepare_data, get_models, fit_model, compute_metrics, best_f1_threshold
        data = prepare_data()
        X_tr, y_tr = data["train"]
        X_va, y_va = data["val"]
        X_te, y_te = data["test"]
        for name, model in get_models().items():
            model = fit_model(name, model, X_tr, y_tr)
            thr = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
            metrics = compute_metrics(y_te, model.predict_proba(X_te)[:, 1], thr)
            ok &= _check(f"baseline {name}  test F1={metrics['f1']:.3f}",
                         metrics["f1"] > 0.5)
    except Exception:
        print(FAIL + "  baselines training")
        traceback.print_exc()
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Workshop 2
# ---------------------------------------------------------------------------

def test_w2() -> bool:
    """Checkpoint, sequence windows (raw, engineered, augmented) and test ROC-AUC."""
    _header("Workshop 2 — LSTM Sequence Model")
    ok = True

    CKPT = ROOT / "checkpoints" / "workshop2" / "model_best.pt"
    if not CKPT.exists():
        print(SKIP + "  checkpoint not found, skipping W2")
        return True

    try:
        import torch
        from w1.dataset import load_raw, load_splits, subset
        from w2.model import build_model
        from w2.sequences import augment_rows, build_all_partitions, compute_features
        from w2.train_lstm import evaluate_loader, make_loaders, set_global_seed

        set_global_seed(42)
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        ok &= _check("checkpoint loaded", "model_state" in ckpt)

        df = load_raw()
        splits = load_splits()
        parts, _ = build_all_partitions(
            df, splits,
            window=int(ckpt["window"]),
            mode=str(ckpt["features"]),
        )
        ok &= _check("build_all_partitions()", len(parts) == 3,
                     f"splits: {list(parts.keys())}")

        # Optional input modes of train_lstm.py (--features engineered, --augment)
        small = subset(df, splits["train"][:5])
        eng = compute_features(small, "engineered")
        ok &= _check("engineered features (--features engineered)", eng.shape[1] == 16,
                     f"{eng.shape[1]} columns")
        aug = augment_rows(small)
        ok &= _check("noise augmentation (--augment)", len(aug) == 2 * len(small),
                     f"{len(small)} -> {len(aug)} rows")

        loaders = make_loaders(parts, batch_size=256, seed=42)
        ok &= _check("make_loaders()", set(loaders.keys()) == {"train", "val", "test"})

        model = build_model(ckpt["model_cfg"])
        model.load_state_dict(ckpt["model_state"])
        ok &= _check("build_model() + load_state_dict()", True)

        crit = torch.nn.BCEWithLogitsLoss()
        y_te, s_te, test_loss = evaluate_loader(
            model, loaders["test"], crit, torch.device("cpu")
        )
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_te, s_te)
        ok &= _check(f"evaluate test set  ROC-AUC={auc:.3f}", auc > 0.80,
                     f"{len(y_te)} samples")

    except Exception:
        print(FAIL + "  W2 pipeline")
        traceback.print_exc()
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Workshop 3
# ---------------------------------------------------------------------------

def test_w3() -> bool:
    """Agent, sweep checkpoints, W1/W2 bridges and the six baseline policies."""
    _header("Workshop 3 — DQN Agent")
    ok = True

    CKPT = ROOT / "checkpoints" / "workshop3" / "agent_best.pt"
    if not CKPT.exists():
        print(SKIP + "  checkpoint not found, skipping W3")
        return True

    try:
        import numpy as np
        from w3.agent import DQNAgent
        from w3.baselines import make_baselines
        from w3.common import RUNS_DIR, load_json
        from w3.env import MAX_STEPS, make_env
        from w3.evaluate import AgentPolicy, run_episode, sweep_checkpoint
        from w3.w1_bridge import W1DangerModel
        from w3.w2_bridge import W2DangerModel

        agent = DQNAgent.load(CKPT)
        ok &= _check("DQNAgent.load()", agent is not None)

        env = make_env()
        policy = AgentPolicy(agent, "best")
        lengths, outcomes = [], []
        for seed in range(5):
            ep = run_episode(policy, env, seed=seed)
            lengths.append(ep["length"])
            outcomes.append(ep["outcome"])
        survival = sum(1 for o in outcomes if o == "time_limit") / len(outcomes)
        mean_len = float(np.mean(lengths))
        ok &= _check(f"5 episodes  survival={survival:.0%}  mean_len={mean_len:.0f}",
                     mean_len > 50 and survival == float(np.mean(np.array(lengths) >= MAX_STEPS)))

        # The sweep checkpoints of the selected configuration (used by evaluate.py)
        sweep = load_json(RUNS_DIR / "sweep_results.json")
        best = [r for r in sweep["runs"] if r["config"] == sweep["best_config"]]
        loaded = [DQNAgent.load(sweep_checkpoint(r)) for r in best]
        ok &= _check(f"sweep checkpoints of '{sweep['best_config']}' load", len(loaded) == 3)

        # Supervised models from Workshops 1 and 2, used as policies and metrics
        w1 = W1DangerModel()
        ok &= _check("W1 classifier rebuilt", 0 < w1.threshold < 1, f"threshold {w1.threshold:.3f}")
        w2 = W2DangerModel()
        ok &= _check("W2 LSTM loaded", 0 < w2.threshold < 1, f"threshold {w2.threshold:.3f}")
        policies = make_baselines(w1, w2_model=w2)
        ok &= _check("make_baselines() with W1 and W2 models", len(policies) == 6,
                     ", ".join(p.name for p in policies))
        ep = run_episode(policies[-1], env, seed=0)          # W2-guided policy
        rate = w2.danger_rate(ep["states"], ep["actions"])
        ok &= _check(f"W2 look-ahead episode  length={ep['length']}  W2 danger={rate:.2%}",
                     ep["length"] > 50)
        env.close()

    except Exception:
        print(FAIL + "  W3 pipeline")
        traceback.print_exc()
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Workshop 4
# ---------------------------------------------------------------------------

def test_w4() -> bool:
    """AE checkpoint, reconstruction, frozen W2 encoder and the W3 monitor imports."""
    _header("Workshop 4 — Autoencoder / VAE")
    ok = True

    CKPT = ROOT / "checkpoints" / "workshop4" / "ae_best.pt"
    if not CKPT.exists():
        print(SKIP + "  checkpoint not found, skipping W4")
        return True

    try:
        import torch
        from w4.w4_common import load_w4_data, reconstruction_metrics
        from w4.autoencoder import build_model, infer

        data = load_w4_data()
        n_train = data.train.X.shape[0]
        ok &= _check("load_w4_data()", n_train > 0, f"train rows: {n_train}")

        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        ok &= _check("ae_best.pt loaded", "state_dict" in ckpt)

        from w4.w4_common import TrainConfig
        cfg = TrainConfig(**ckpt["config"])
        input_dim = ckpt.get("input_dim", 5)
        model = build_model(cfg, input_dim=input_dim)
        model.load_state_dict(ckpt["state_dict"])
        ok &= _check("build_model() + load_state_dict()", True)

        X_test = data.test.X
        out = infer(model, X_test)
        recon_key = "recon" if "recon" in out else "X_hat"
        ok &= _check("infer() on the test partition", recon_key in out,
                     f"recon shape {out[recon_key].shape}")

        metrics = reconstruction_metrics(X_test, out[recon_key])
        ok &= _check(f"reconstruction_metrics()  MSE={metrics['mse']:.4f}",
                     metrics["mse"] < 10.0)

        # Frozen Workshop 2 encoder (used by evaluate_ae.py and analyze_latent.py)
        from w4.w2_encoder import load_w2
        from w4.w4_common import w2_windows
        _, meta = load_w2()
        X, lengths, episode, _t = w2_windows(meta, "test")
        ok &= _check("W2 windows for the test partition", len(X) == len(data.test.X),
                     f"{X.shape}")

        # Label-free monitoring of the Workshop 3 controllers (monitor_w3.py)
        import w4.monitor_w3  # noqa: F401
        ok &= _check("monitor_w3 imports (W3 policies + W1/W2 bridges)", True)

    except Exception:
        print(FAIL + "  W4 pipeline")
        traceback.print_exc()
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = {1: test_w1, 2: test_w2, 3: test_w3, 4: test_w4}


def main() -> None:
    """Run the repository checks and the selected workshop tests."""
    p = argparse.ArgumentParser(description="Integration tests for all workshops")
    p.add_argument("--w", nargs="+", type=int, default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4], metavar="N", help="workshops to test")
    args = p.parse_args()

    results: dict[str, bool] = {}
    t0 = time.time()
    results["Repository"] = test_repo()
    for n in args.w:
        results[f"Workshop {n}"] = TESTS[n]()

    elapsed = time.time() - t0
    _header(f"Summary  ({elapsed:.1f}s)")
    all_ok = True
    for name, passed in results.items():
        icon = PASS if passed else FAIL
        print(f"{icon}  {name}")
        all_ok &= passed

    print()
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
