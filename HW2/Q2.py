import math
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from the_well.data import WellDataset

import matplotlib.pyplot as plt


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. Config
# =========================================================
@dataclass
class Config:
    # Remote reading from Hugging Face
    base_path: str = "hf://datasets/polymathic-ai/"
    dataset_name: str = "gray_scott_reaction_diffusion"

    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"

    # One-step prediction
    n_steps_input: int = 1
    n_steps_output: int = 1

    # Lightweight training settings
    batch_size: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 3
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Small subsets only
    max_train_samples: int | None = 80
    max_valid_samples: int | None = 20
    max_test_samples: int | None = 20

    # Model size
    hidden_channels: int = 16
    kernel_size: int = 3

    # Translation sanity check
    shift_x: int = 5
    shift_y: int = 7


# =========================================================
# 3. Dataset utilities
# =========================================================
def build_dataset(
    base_path: str,
    dataset_name: str,
    split_name: str,
    n_steps_input: int,
    n_steps_output: int,
):
    dataset = WellDataset(
        well_base_path=base_path,
        well_dataset_name=dataset_name,
        well_split_name=split_name,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        use_normalization=False,
    )
    return dataset


def maybe_subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def fields_to_nchw(batch_fields: torch.Tensor) -> torch.Tensor:
    """
    Convert The Well field tensor to PyTorch Conv2d format.

    Expected shape:
        (B, T, H, W, F)

    For our one-step setup, T = 1, so:
        (B, 1, H, W, F) -> (B, F, H, W)
    """
    if batch_fields.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B, T, H, W, F), got {batch_fields.shape}")

    if batch_fields.shape[1] != 1:
        raise ValueError(
            f"This code assumes one-step input/output (T=1). Got T={batch_fields.shape[1]}"
        )

    x = batch_fields[:, 0]                  # (B, H, W, F)
    x = x.permute(0, 3, 1, 2).contiguous()  # (B, F, H, W)
    return x.float()


# =========================================================
# 4. Models
# =========================================================
class SimpleCNN(nn.Module):
    """
    Fully convolutional network for one-step prediction.

    padding_mode:
        "circular" -> constrained model
        "zeros"    -> baseline
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding_mode: str = "circular",
    ):
        super().__init__()
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# 5. Evaluation
# =========================================================
@torch.no_grad()
def evaluate_model(model, loader, device) -> Dict[str, float]:
    model.eval()

    mse_sum = 0.0
    n_batches = 0

    for batch in loader:
        x = fields_to_nchw(batch["input_fields"]).to(device)
        y = fields_to_nchw(batch["output_fields"]).to(device)

        pred = model(x)
        mse = torch.mean((pred - y) ** 2)

        mse_sum += mse.item()
        n_batches += 1

    return {
        "mse": mse_sum / max(n_batches, 1),
    }


@torch.no_grad()
def translation_equivariance_error(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str,
    shift_x: int,
    shift_y: int,
) -> float:
    """
    Check whether:
        f(Tx) ≈ T(f(x))
    """
    model.eval()

    x = fields_to_nchw(batch["input_fields"]).to(device)

    shifted_x = torch.roll(x, shifts=(shift_x, shift_y), dims=(-2, -1))

    lhs = model(shifted_x)
    rhs = torch.roll(model(x), shifts=(shift_x, shift_y), dims=(-2, -1))

    rel_err = torch.norm(lhs - rhs) / (torch.norm(rhs) + 1e-12)
    return rel_err.item()


# =========================================================
# 6. Training
# =========================================================
def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = fields_to_nchw(batch["input_fields"]).to(device)
        y = fields_to_nchw(batch["output_fields"]).to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_training(model, train_loader, valid_loader, cfg: Config, model_name: str):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_valid_mse = math.inf
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        valid_metrics = evaluate_model(model, valid_loader, cfg.device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_mse": valid_metrics["mse"],
        })

        print(
            f"[{model_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"valid_mse={valid_metrics['mse']:.6f}"
        )

        if valid_metrics["mse"] < best_valid_mse:
            best_valid_mse = valid_metrics["mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

@torch.no_grad()
def plot_predictions_separately(constrained_model, baseline_model, test_loader, device):
    constrained_model.eval()
    baseline_model.eval()

    batch = next(iter(test_loader))
    x = fields_to_nchw(batch["input_fields"]).to(device)
    y = fields_to_nchw(batch["output_fields"]).to(device)

    pred_c = constrained_model(x)
    pred_b = baseline_model(x)

    # first sample, first channel
    y0 = y[0, 0].detach().cpu().numpy()
    pc0 = pred_c[0, 0].detach().cpu().numpy()
    pb0 = pred_b[0, 0].detach().cpu().numpy()

    vmin = min(y0.min(), pc0.min(), pb0.min())
    vmax = max(y0.max(), pc0.max(), pb0.max())

    plt.figure(figsize=(6, 5))
    plt.imshow(y0, vmin=vmin, vmax=vmax, origin="lower")
    plt.title("Ground Truth")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Field value")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(pc0, vmin=vmin, vmax=vmax, origin="lower")
    plt.title("Constrained Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Field value")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(pb0, vmin=vmin, vmax=vmax, origin="lower")
    plt.title("Baseline Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Field value")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_errors_separately(constrained_model, baseline_model, test_loader, device):
    constrained_model.eval()
    baseline_model.eval()

    batch = next(iter(test_loader))
    x = fields_to_nchw(batch["input_fields"]).to(device)
    y = fields_to_nchw(batch["output_fields"]).to(device)

    pred_c = constrained_model(x)
    pred_b = baseline_model(x)

    y0 = y[0, 0].detach().cpu().numpy()
    pc0 = pred_c[0, 0].detach().cpu().numpy()
    pb0 = pred_b[0, 0].detach().cpu().numpy()

    err_c = np.abs(pc0 - y0)
    err_b = np.abs(pb0 - y0)

    vmin = min(err_c.min(), err_b.min())
    vmax = max(err_c.max(), err_b.max())

    plt.figure(figsize=(6, 5))
    plt.imshow(err_c, vmin=vmin, vmax=vmax, origin="lower")
    plt.title("Constrained Absolute Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Absolute error")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(err_b, vmin=vmin, vmax=vmax, origin="lower")
    plt.title("Baseline Absolute Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Absolute error")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_linecut_comparison(constrained_model, baseline_model, test_loader, device):
    constrained_model.eval()
    baseline_model.eval()

    batch = next(iter(test_loader))
    x = fields_to_nchw(batch["input_fields"]).to(device)
    y = fields_to_nchw(batch["output_fields"]).to(device)

    pred_c = constrained_model(x)
    pred_b = baseline_model(x)

    # first sample, first channel
    y0 = y[0, 0].detach().cpu().numpy()
    pc0 = pred_c[0, 0].detach().cpu().numpy()
    pb0 = pred_b[0, 0].detach().cpu().numpy()

    # horizontal center line
    mid_row = y0.shape[0] // 2
    x_axis = np.arange(y0.shape[1])

    gt_line = y0[mid_row, :]
    c_line = pc0[mid_row, :]
    b_line = pb0[mid_row, :]

    plt.figure(figsize=(7, 5))
    plt.plot(x_axis, gt_line, label="Ground Truth")
    plt.plot(x_axis, c_line, label="Constrained Prediction")
    plt.plot(x_axis, b_line, label="Baseline Prediction")
    plt.xlabel("x")
    plt.ylabel("Field value")
    plt.title(f"Linecut Comparison at y = {mid_row}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================================================
# 7. Main
# =========================================================
def main():
    cfg = Config()
    set_seed(42)

    print(f"Using device: {cfg.device}")

    # Build datasets
    train_dataset = build_dataset(
        cfg.base_path,
        cfg.dataset_name,
        cfg.train_split,
        cfg.n_steps_input,
        cfg.n_steps_output,
    )
    valid_dataset = build_dataset(
        cfg.base_path,
        cfg.dataset_name,
        cfg.valid_split,
        cfg.n_steps_input,
        cfg.n_steps_output,
    )
    test_dataset = build_dataset(
        cfg.base_path,
        cfg.dataset_name,
        cfg.test_split,
        cfg.n_steps_input,
        cfg.n_steps_output,
    )

    # Use only small subsets
    train_dataset = maybe_subset(train_dataset, cfg.max_train_samples)
    valid_dataset = maybe_subset(valid_dataset, cfg.max_valid_samples)
    test_dataset = maybe_subset(test_dataset, cfg.max_test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Infer channels from a sample batch
    sample = next(iter(train_loader))
    x0 = fields_to_nchw(sample["input_fields"])
    y0 = fields_to_nchw(sample["output_fields"])

    in_channels = x0.shape[1]
    out_channels = y0.shape[1]

    print(f"in_channels={in_channels}, out_channels={out_channels}")
    print(f"input shape after conversion: {tuple(x0.shape)}")

    # Constrained model
    constrained_model = SimpleCNN(
        in_channels=in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=out_channels,
        kernel_size=cfg.kernel_size,
        padding_mode="circular",
    ).to(cfg.device)

    # Baseline model
    baseline_model = SimpleCNN(
        in_channels=in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=out_channels,
        kernel_size=cfg.kernel_size,
        padding_mode="zeros",
    ).to(cfg.device)

    # Train constrained model
    print("\nTraining constrained model (circular padding)...")
    constrained_history = run_training(
        constrained_model,
        train_loader,
        valid_loader,
        cfg,
        model_name="Constrained",
    )

    # Train baseline model
    print("\nTraining baseline model (zero padding)...")
    baseline_history = run_training(
        baseline_model,
        train_loader,
        valid_loader,
        cfg,
        model_name="Baseline",
    )

    # Final test metrics
    constrained_test = evaluate_model(constrained_model, test_loader, cfg.device)
    baseline_test = evaluate_model(baseline_model, test_loader, cfg.device)

    print("\n=== Final Test Results ===")
    print(f"Constrained | test_mse={constrained_test['mse']:.6f}")
    print(f"Baseline    | test_mse={baseline_test['mse']:.6f}")

    # Sanity check
    sanity_batch = next(iter(test_loader))

    constrained_eq_err = translation_equivariance_error(
        constrained_model,
        sanity_batch,
        cfg.device,
        cfg.shift_x,
        cfg.shift_y,
    )
    baseline_eq_err = translation_equivariance_error(
        baseline_model,
        sanity_batch,
        cfg.device,
        cfg.shift_x,
        cfg.shift_y,
    )

    print("\n=== Translation Equivariance Sanity Check ===")
    print(f"Constrained relative error: {constrained_eq_err:.6e}")
    print(f"Baseline    relative error: {baseline_eq_err:.6e}")

    print("\nDone.")
    plot_predictions_separately(constrained_model, baseline_model, test_loader, cfg.device)
    plot_errors_separately(constrained_model, baseline_model, test_loader, cfg.device)
    plot_linecut_comparison(constrained_model, baseline_model, test_loader, cfg.device)
    



if __name__ == "__main__":
    main()
    
    