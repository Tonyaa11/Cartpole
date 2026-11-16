"""
integration_test.py  —  End-to-end pipeline test for all four workshops.

Each workshop test loads real data/checkpoints and runs the full pipeline at
minimal scale (a few epochs / episodes) to catch import errors, shape mismatches,
and broken file paths before submitting.

Usage
-----
    python scripts/integration_test.py           # all workshops
    python scripts/integration_test.py --w 1 2  # specific workshops
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PASS = "  PASS"
FAIL = "  FAIL"
SKIP = "  SKIP"


def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _check(label: str, cond: bool, detail: str = "") -> bool:
    icon = PASS if cond else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{icon}  {label}{suffix}")
    return cond


# ---------------------------------------------------------------------------
# Workshop 1
# ---------------------------------------------------------------------------

def test_w1() -> bool:
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

    # 3. Train all three baselines end-to-end
    try:
        from w1.baselines import prepare_data, get_models, fit_model, compute_metrics, best_f1_threshold
        import numpy as np
        data = prepare_data()
        X_tr, y_tr = data["train"]
        X_te, y_te = data["test"]
        results = {}
        for name, model in get_models().items():
            model = fit_model(name, model, X_tr, y_tr)
            scores = model.predict_proba(X_te)[:, 1]
            thr = best_f1_threshold(y_te, scores)
            metrics = compute_metrics(y_te, scores, thr)
            results[name] = metrics
            ok &= _check(f"baseline {name}  F1={metrics['f1']:.3f}",
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
    _header("Workshop 2 — LSTM Sequence Model")
    ok = True

    CKPT = ROOT / "checkpoints" / "workshop2" / "model_best.pt"
    if not CKPT.exists():
        print(SKIP + "  checkpoint not found, skipping W2")
        return True

    try:
        import torch
        from w1.dataset import load_raw, load_splits
        from w2.model import build_model
        from w2.sequences import build_all_partitions
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
    _header("Workshop 3 — DQN Agent")
    ok = True

    CKPT = ROOT / "checkpoints" / "workshop3" / "agent_best.pt"
    if not CKPT.exists():
        print(SKIP + "  checkpoint not found, skipping W3")
        return True

    try:
        import numpy as np
        from w3.agent import DQNAgent
        from w3.env import make_env
        from w3.evaluate import AgentPolicy, run_episode

        agent = DQNAgent.load(CKPT)
        ok &= _check("DQNAgent.load()", agent is not None)

        env = make_env()
        policy = AgentPolicy(agent, "best")

        lengths = []
        outcomes = []
        for seed in range(5):
            ep = run_episode(policy, env, seed=seed)
            lengths.append(ep["length"])
            outcomes.append(ep["outcome"])

        env.close()
        survival = sum(1 for o in outcomes if o == "survived") / len(outcomes)
        mean_len = float(np.mean(lengths))
        ok &= _check(f"5 episodes  survival={survival:.0%}  mean_len={mean_len:.0f}",
                     mean_len > 50)

    except Exception:
        print(FAIL + "  W3 pipeline")
        traceback.print_exc()
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Workshop 4
# ---------------------------------------------------------------------------

def test_w4() -> bool:
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

        X_test = data.training_matrix("normal")
        out = infer(model, X_test)
        recon_key = "recon" if "recon" in out else "X_hat"
        ok &= _check("infer() on test data", recon_key in out,
                     f"recon shape {out[recon_key].shape}")

        metrics = reconstruction_metrics(X_test, out[recon_key])
        ok &= _check(f"reconstruction_metrics()  MSE={metrics['mse']:.4f}",
                     metrics["mse"] < 10.0)

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
    p = argparse.ArgumentParser(description="Integration tests for all workshops")
    p.add_argument("--w", nargs="+", type=int, default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4], metavar="N", help="workshops to test")
    args = p.parse_args()

    results: dict[int, bool] = {}
    t0 = time.time()
    for n in args.w:
        results[n] = TESTS[n]()

    elapsed = time.time() - t0
    _header(f"Summary  ({elapsed:.1f}s)")
    all_ok = True
    for n, passed in results.items():
        icon = PASS if passed else FAIL
        print(f"{icon}  Workshop {n}")
        all_ok &= passed

    print()
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
