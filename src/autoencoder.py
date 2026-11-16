"""
Autoencoder (AE), Variational Autoencoder (VAE) and supervised reference MLP.

Machine Learning - Workshop 4 (Autoencoders & Variational Autoencoders)

The CartPole data are TABULAR (16 standardised engineered features per time
step), so every network is built from fully-connected layers, as the workshop
recommends for tabular data:

    Encoder : 16 -> 64 -> 32 -> z            (ReLU)
    Decoder : z  -> 32 -> 64 -> 16           (ReLU, LINEAR output)

* The output layer is linear because the targets are z-scored real numbers
  (unbounded, roughly N(0, 1)); a Sigmoid/Tanh would clip them.
* **AE** loss: mean squared error (MSE).
* **VAE**: the encoder has two heads, ``mu`` and ``log sigma^2``. A latent
  sample is drawn with the *reparameterisation trick*
  ``z = mu + sigma * eps, eps ~ N(0, I)``, which moves the randomness to an
  input (eps) so that gradients flow through ``mu`` and ``sigma``.
  Loss (negative ELBO):
      L = sum_j (x_j - x_hat_j)^2  +  beta * KL( q(z|x) || N(0, I) )
      KL = -1/2 * sum_k (1 + log sigma_k^2 - mu_k^2 - sigma_k^2)
  The reconstruction term is the squared error summed over the 16 features
  (Gaussian decoder with fixed variance), so it is on the same per-sample scale
  as the KL term. KL annealing ramps beta linearly from 0 to its target value
  during the first ``kl_warmup_epochs`` epochs to avoid posterior collapse.
* **Denoising AE** (optional extension, Vincent et al. 2010): same AE, but the
  encoder receives ``x + noise`` and the loss compares with the clean ``x``.
* **Supervised MLP control**: same encoder trunk with a bottleneck of the same
  width, followed by a logistic output trained with the labels. It isolates the
  effect of the labels (same input, same architecture) in the latent analysis;
  the Workshop 2 deep LSTM is the main supervised reference (``w2_encoder.py``).

Everything that touches PyTorch lives in this module; the experiment logic is
in ``train_ae.py`` and the framework-independent utilities in ``w4_common.py``.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from w4_common import TrainConfig

LOGVAR_CLAMP = (-10.0, 10.0)  # numerical safety for exp(logvar)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Fix every random number generator used during training.

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def mlp(dims: List[int], final_activation: bool) -> nn.Sequential:
    """Stack of Linear + ReLU layers.

    Args:
        dims (List[int]): Layer widths, e.g. ``[16, 64, 32]``.
        final_activation (bool): Whether to apply ReLU after the last Linear.

    Returns:
        nn.Sequential: The network.
    """
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Deterministic fully-connected autoencoder.

    Args:
        input_dim (int): Number of input features (16).
        hidden (List[int]): Encoder hidden widths (decoder mirrors them).
        z_dim (int): Bottleneck width.
    """

    def __init__(self, input_dim: int, hidden: List[int], z_dim: int) -> None:
        super().__init__()
        self.encoder = mlp([input_dim, *hidden, z_dim], final_activation=False)
        self.decoder = mlp([z_dim, *reversed(hidden), input_dim], final_activation=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map inputs to the latent code."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent codes back to the input space (linear output)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        return {"recon": self.decode(z), "z": z}


class VariationalAutoencoder(nn.Module):
    """Gaussian VAE with a standard-normal prior N(0, I).

    Args:
        input_dim (int): Number of input features (16).
        hidden (List[int]): Encoder hidden widths (decoder mirrors them).
        z_dim (int): Latent dimension.
    """

    def __init__(self, input_dim: int, hidden: List[int], z_dim: int) -> None:
        super().__init__()
        self.trunk = mlp([input_dim, *hidden], final_activation=True)
        self.fc_mu = nn.Linear(hidden[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden[-1], z_dim)
        self.decoder = mlp([z_dim, *reversed(hidden), input_dim], final_activation=False)
        self.z_dim = z_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the parameters (mu, log sigma^2) of q(z | x)."""
        h = self.trunk(x)
        logvar = torch.clamp(self.fc_logvar(h), *LOGVAR_CLAMP)
        return self.fc_mu(h), logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mu + sigma * eps, eps ~ N(0, I).

        The sample is a deterministic, differentiable function of (mu, sigma);
        the randomness enters only through eps.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent codes back to the input space (linear output)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        # Training: stochastic sample. Evaluation: use the posterior mean.
        z = self.reparameterize(mu, logvar) if self.training else mu
        return {"recon": self.decode(z), "z": z, "mu": mu, "logvar": logvar}


class SupervisedMLP(nn.Module):
    """Supervised failure classifier used as a same-architecture control.

    The encoder trunk mirrors the autoencoders (16 -> 64 -> 32 -> z, ReLU) so
    that the bottleneck activations are directly comparable with the
    unsupervised latent codes.

    Args:
        input_dim (int): Number of input features (16).
        hidden (List[int]): Hidden widths.
        z_dim (int): Width of the representation layer.
    """

    def __init__(self, input_dim: int, hidden: List[int], z_dim: int) -> None:
        super().__init__()
        self.encoder = mlp([input_dim, *hidden, z_dim], final_activation=True)
        self.head = nn.Linear(z_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen supervised representation (bottleneck activations)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        return {"logit": self.head(z).squeeze(-1), "z": z}


def build_model(cfg: TrainConfig, input_dim: int) -> nn.Module:
    """Instantiate the network described by a configuration.

    Args:
        cfg (TrainConfig): Run configuration.
        input_dim (int): Number of input features.

    Returns:
        nn.Module: Untrained model.
    """
    if cfg.model == "ae":
        return Autoencoder(input_dim, cfg.hidden, cfg.z_dim)
    if cfg.model == "vae":
        return VariationalAutoencoder(input_dim, cfg.hidden, cfg.z_dim)
    if cfg.model == "sup":
        return SupervisedMLP(input_dim, cfg.hidden, cfg.z_dim)
    raise ValueError(f"Unknown model type: {cfg.model}")


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def sse_per_sample(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Squared error summed over features, one value per sample."""
    return ((x - x_hat) ** 2).sum(dim=1)


def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL( N(mu, sigma^2) || N(0, I) ), one value per sample."""
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_weight(cfg: TrainConfig, epoch: int) -> float:
    """KL annealing schedule: linear ramp from 0 to beta over the warm-up epochs.

    Args:
        cfg (TrainConfig): Run configuration.
        epoch (int): Zero-based epoch index.

    Returns:
        float: Effective beta for this epoch.
    """
    if cfg.kl_warmup_epochs <= 0:
        return cfg.beta
    return cfg.beta * min(1.0, epoch / cfg.kl_warmup_epochs)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
LogFn = Callable[[int, Dict[str, float]], None]


def _run_epoch_unsupervised(
    model: nn.Module, cfg: TrainConfig, loader: DataLoader, beta: float,
    optimizer: Optional[torch.optim.Optimizer], gen: torch.Generator,
) -> Dict[str, float]:
    """One pass over a loader (training when an optimizer is given).

    Returns:
        Dict[str, float]: loss (objective with the current beta), recon_mse,
        kl and loss_target (objective with the target beta).
    """
    train = optimizer is not None
    model.train(train)
    tot = {"loss": 0.0, "recon_mse": 0.0, "kl": 0.0, "loss_target": 0.0}
    n_total = 0
    for (x_clean,) in loader:
        x_in = x_clean
        if cfg.noise_std > 0:  # denoising: corrupt the input, keep a clean target
            noise = torch.randn(x_clean.shape, generator=gen) * cfg.noise_std
            x_in = x_clean + noise
        with torch.set_grad_enabled(train):
            out = model(x_in)
            sse = sse_per_sample(x_clean, out["recon"])
            if cfg.model == "vae":
                kl = kl_per_sample(out["mu"], out["logvar"])
                loss = (sse + beta * kl).mean()
            else:
                kl = torch.zeros_like(sse)
                loss = sse.mean() / x_clean.shape[1]  # plain MSE
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        n = x_clean.shape[0]
        n_total += n
        d = x_clean.shape[1]
        tot["loss"] += float(loss) * n
        tot["recon_mse"] += float(sse.sum()) / d
        tot["kl"] += float(kl.sum())
        target = (sse + cfg.beta * kl).mean() if cfg.model == "vae" else sse.mean() / d
        tot["loss_target"] += float(target) * n
    return {k: v / n_total for k, v in tot.items()}


def fit(
    cfg: TrainConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    log_fn: Optional[LogFn] = None,
) -> Tuple[nn.Module, List[Dict[str, float]], int]:
    """Train one model with Adam and keep the best epoch on validation.

    Checkpoint selection:
        * AE / DAE: lowest validation MSE.
        * VAE: lowest validation loss with the TARGET beta, only after the KL
          warm-up has finished (earlier epochs optimise a different objective).
        * Supervised MLP: highest validation PR-AUC (imbalanced labels).

    Args:
        cfg (TrainConfig): Run configuration.
        X_train (np.ndarray): Training inputs (standardised).
        X_val (np.ndarray): Validation inputs (standardised).
        y_train (Optional[np.ndarray]): Training labels (supervised model only).
        y_val (Optional[np.ndarray]): Validation labels (supervised model only).
        log_fn (Optional[LogFn]): Called after every epoch with the metrics.

    Returns:
        Tuple[nn.Module, List[Dict[str, float]], int]: best model (weights of
        the best epoch), per-epoch history, best epoch index.
    """
    seed_everything(cfg.seed)
    model = build_model(cfg, X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loader_gen = torch.Generator().manual_seed(cfg.seed)
    noise_gen = torch.Generator().manual_seed(cfg.seed + 1)
    Xtr = torch.as_tensor(X_train, dtype=torch.float32)
    Xva = torch.as_tensor(X_val, dtype=torch.float32)

    if cfg.model == "sup":
        ytr = torch.as_tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size,
                                  shuffle=True, generator=loader_gen)
        n_pos = max(float(y_train.sum()), 1.0)
        pos_weight = torch.tensor((len(y_train) - n_pos) / n_pos, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        train_loader = DataLoader(TensorDataset(Xtr), batch_size=cfg.batch_size,
                                  shuffle=True, generator=loader_gen)
        # Fixed corruption of the validation set for the denoising AE
        if cfg.noise_std > 0:
            val_gen = torch.Generator().manual_seed(cfg.seed + 2)
            Xva_in = Xva + torch.randn(Xva.shape, generator=val_gen) * cfg.noise_std
        else:
            Xva_in = Xva

    history: List[Dict[str, float]] = []
    best_state, best_epoch, best_score = None, -1, None

    for epoch in range(cfg.epochs):
        t0 = time.time()
        if cfg.model == "sup":
            model.train()
            tot, n_tot = 0.0, 0
            for xb, yb in train_loader:
                loss = criterion(model(xb)["logit"], yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot += float(loss) * len(xb)
                n_tot += len(xb)
            model.eval()
            with torch.no_grad():
                val_logit = model(Xva)["logit"]
                val_loss = float(criterion(val_logit, torch.as_tensor(y_val, dtype=torch.float32)))
                p = torch.sigmoid(val_logit).numpy()
            rec = {
                "epoch": epoch, "train_loss": tot / n_tot, "val_loss": val_loss,
                "val_roc_auc": float(roc_auc_score(y_val, p)),
                "val_pr_auc": float(average_precision_score(y_val, p)),
            }
            score, eligible = -rec["val_pr_auc"], True
        else:
            beta = kl_weight(cfg, epoch) if cfg.model == "vae" else 0.0
            tr = _run_epoch_unsupervised(model, cfg, train_loader, beta, optimizer, noise_gen)
            # Validation (deterministic: VAE uses mu in eval mode, DAE fixed noise)
            model.eval()
            with torch.no_grad():
                out = model(Xva_in)
                sse = sse_per_sample(Xva, out["recon"])
                kl = (kl_per_sample(out["mu"], out["logvar"]) if cfg.model == "vae"
                      else torch.zeros_like(sse))
            d = Xva.shape[1]
            va = {
                "recon_mse": float(sse.mean()) / d,
                "kl": float(kl.mean()),
                "loss": float((sse + beta * kl).mean()) if cfg.model == "vae" else float(sse.mean()) / d,
                "loss_target": (float((sse + cfg.beta * kl).mean()) if cfg.model == "vae"
                                else float(sse.mean()) / d),
            }
            rec = {"epoch": epoch, "beta_eff": beta}
            rec.update({f"train_{k}": v for k, v in tr.items()})
            rec.update({f"val_{k}": v for k, v in va.items()})
            score = va["loss_target"]
            eligible = cfg.model != "vae" or epoch >= min(cfg.kl_warmup_epochs, cfg.epochs - 1)
        rec["epoch_time_s"] = time.time() - t0
        history.append(rec)
        if log_fn is not None:
            log_fn(epoch, rec)

        if eligible and (best_score is None or score < best_score):
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


# --------------------------------------------------------------------------- #
# Inference (NumPy in, NumPy out)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def infer(model: nn.Module, X: np.ndarray, noise_std: float = 0.0, seed: int = 0) -> Dict[str, np.ndarray]:
    """Run a trained model on a NumPy array.

    Returns, depending on the model:
        * AE: ``z`` (latent code) and ``recon``.
        * VAE: ``z`` (= posterior mean mu), ``logvar``, ``recon`` (decoded from
          mu, deterministic) and ``kl`` (per-sample KL divergence).
        * Supervised MLP: ``z`` (frozen representation) and ``proba``.

    Args:
        model (nn.Module): Trained model.
        X (np.ndarray): Standardised inputs.
        noise_std (float): Optional Gaussian corruption of the inputs (used to
            evaluate denoising); 0 = clean inputs.
        seed (int): Seed of the corruption noise.

    Returns:
        Dict[str, np.ndarray]: Outputs as NumPy arrays.
    """
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32)
    if noise_std > 0:
        x = x + torch.randn(x.shape, generator=torch.Generator().manual_seed(seed)) * noise_std
    if isinstance(model, VariationalAutoencoder):
        mu, logvar = model.encode(x)
        return {
            "z": mu.numpy(),
            "logvar": logvar.numpy(),
            "recon": model.decode(mu).numpy(),
            "kl": kl_per_sample(mu, logvar).numpy(),
        }
    if isinstance(model, SupervisedMLP):
        out = model(x)
        return {"z": out["z"].numpy(), "proba": torch.sigmoid(out["logit"]).numpy()}
    z = model.encode(x)
    return {"z": z.numpy(), "recon": model.decode(z).numpy()}


@torch.no_grad()
def decode(model: nn.Module, Z: np.ndarray) -> np.ndarray:
    """Decode arbitrary latent vectors (AE or VAE) to the input space.

    Args:
        model (nn.Module): Trained AE or VAE.
        Z (np.ndarray): Latent vectors, shape (n, z_dim).

    Returns:
        np.ndarray: Decoded standardised features.
    """
    model.eval()
    return model.decode(torch.as_tensor(Z, dtype=torch.float32)).numpy()


@torch.no_grad()
def generate(model: VariationalAutoencoder, n: int, seed: int = 0) -> np.ndarray:
    """Sample new data from the VAE: z ~ N(0, I), x = decoder(z).

    Args:
        model (VariationalAutoencoder): Trained VAE.
        n (int): Number of samples.
        seed (int): Seed of the latent samples.

    Returns:
        np.ndarray: Generated standardised feature vectors, shape (n, 16).
    """
    model.eval()
    z = torch.randn((n, model.z_dim), generator=torch.Generator().manual_seed(seed))
    return model.decode(z).numpy()


# --------------------------------------------------------------------------- #
# Checkpoints and logging
# --------------------------------------------------------------------------- #
def save_checkpoint(model: nn.Module, cfg: TrainConfig, input_dim: int, path: Path,
                    extra: Optional[Dict] = None) -> None:
    """Save weights + configuration so the model can be rebuilt without code changes.

    Args:
        model (nn.Module): Trained model.
        cfg (TrainConfig): Its configuration.
        input_dim (int): Number of input features.
        path (Path): Destination ``.pt`` file.
        extra (Optional[Dict]): Additional metadata (best epoch, metrics...).

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": cfg.to_dict(),
        "input_dim": input_dim,
        "state_dict": model.state_dict(),
        "extra": extra or {},
        "framework": f"pytorch {torch.__version__}",
    }, path)


def load_checkpoint(path: Path) -> Tuple[nn.Module, TrainConfig, Dict]:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`.

    Args:
        path (Path): ``.pt`` file.

    Returns:
        Tuple[nn.Module, TrainConfig, Dict]: model (eval mode), config, metadata.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg, ckpt["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt.get("extra", {})


def make_tb_writer(logdir: Path):
    """Create a TensorBoard writer if TensorBoard is installed, otherwise None.

    Args:
        logdir (Path): Log directory (``runs/workshop4/<run>/tb``).

    Returns:
        Optional[SummaryWriter]: Writer or None.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # tensorboard not installed: CSV/JSON logs are still written
        return None
    return SummaryWriter(log_dir=str(logdir))
