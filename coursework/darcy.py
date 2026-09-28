from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LpLoss:
    def __init__(self, d: int = 2, p: int = 2, reduction: bool = True, size_average: bool = True) -> None:
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        h = 1.0 / (x.size(1) - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, dim=1
        )
        return self._reduce(all_norms)

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, dim=1)
        all_norms = diff_norms / y_norms
        return self._reduce(all_norms)

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        if not self.reduction:
            return values
        if self.size_average:
            return torch.mean(values)
        return torch.sum(values)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)


class MatReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.path, "r") as handle:
            a_field = np.array(handle["a_field"]).T
            u_field = np.array(handle["u_field"]).T
        return torch.tensor(a_field, dtype=torch.float32), torch.tensor(u_field, dtype=torch.float32)


class UnitGaussianNormalizer:
    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean


@dataclass
class DarcyConfig:
    train_path: Path
    test_path: Path
    output_dir: Path
    batch_size: int = 20
    epochs: int = 200
    learning_rate: float = 1e-3
    scheduler_step: int = 50
    scheduler_gamma: float = 0.7
    seed: int = 7
    sample_index: int = 0
    checkpoint_interval: int = 25
    history_interval: int = 5
    resume_from: Path | None = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DarcyCNN(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.unsqueeze(1)).squeeze(1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class ChannelMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DarcyFNO(nn.Module):
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.mlp0 = ChannelMLP(width, width, width)
        self.mlp1 = ChannelMLP(width, width, width)
        self.mlp2 = ChannelMLP(width, width, width)
        self.mlp3 = ChannelMLP(width, width, width)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.output_proj = ChannelMLP(width, 1, width * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        x = self.input_proj(x).permute(0, 3, 1, 2)

        x = F.gelu(self.mlp0(self.conv0(x)) + self.w0(x))
        x = F.gelu(self.mlp1(self.conv1(x)) + self.w1(x))
        x = F.gelu(self.mlp2(self.conv2(x)) + self.w2(x))
        x = self.mlp3(self.conv3(x)) + self.w3(x)

        return self.output_proj(x).squeeze(1)

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        grid_x = torch.linspace(0, 1, size_x, device=device).view(1, size_x, 1, 1)
        grid_y = torch.linspace(0, 1, size_y, device=device).view(1, 1, size_y, 1)
        grid_x = grid_x.repeat(batch_size, 1, size_y, 1)
        grid_y = grid_y.repeat(batch_size, size_x, 1, 1)
        return torch.cat((grid_x, grid_y), dim=-1)


def load_darcy_data(train_path: Path, test_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_a, train_u = MatReader(train_path).read()
    test_a, test_u = MatReader(test_path).read()
    return train_a, train_u, test_a, test_u


def plot_loss_curves(train_history: list[float], test_history: list[float], output_path: Path) -> None:
    epochs = np.arange(1, len(train_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_history, label="Train loss")
    plt.plot(epochs, test_history, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prediction_comparison(
    truth: torch.Tensor, pred: torch.Tensor, coefficient: torch.Tensor, output_path: Path
) -> None:
    truth_np = truth.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    coeff_np = coefficient.detach().cpu().numpy()
    error_np = np.abs(pred_np - truth_np)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        (coeff_np, truth_np, pred_np, error_np),
        ("Coefficient a(x)", "Truth u(x)", "Prediction u(x)", "|error|"),
        strict=True,
    ):
        im = ax.contourf(field, levels=30, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_loader(a_train: torch.Tensor, u_train: torch.Tensor, batch_size: int) -> data.DataLoader:
    train_set = data.TensorDataset(a_train, u_train)
    return data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


def save_training_state(
    output_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    history: dict[str, list[float]],
    config: DarcyConfig,
    device: torch.device,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "config": config.__dict__,
            "device": str(device),
        },
        output_path,
    )


def load_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, dict[str, list[float]]]:
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    history = state.get("history", {"train": [], "test": []})
    return int(state.get("epoch", 0)), history


def run_training(config: DarcyConfig, model: nn.Module) -> dict[str, Path]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    print(f"Using device: {device}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    a_train, u_train, a_test, u_test = load_darcy_data(config.train_path, config.test_path)
    model = model.to(device)

    normalizer = UnitGaussianNormalizer(a_train)
    a_train_norm = normalizer.encode(a_train)
    a_test_norm = normalizer.encode(a_test)

    train_loader = _build_loader(a_train_norm, u_train, config.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    loss_fn = LpLoss()

    train_history: list[float] = []
    test_history: list[float] = []
    checkpoint_path = config.output_dir / "checkpoint_latest.pt"
    history_path = config.output_dir / "training_history.csv"
    start_epoch = 0
    if config.resume_from is not None:
        start_epoch, history = load_training_state(config.resume_from, model, optimizer, scheduler, device)
        train_history = history.get("train", [])
        test_history = history.get("test", [])
        print(f"Resuming from checkpoint: {config.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch + 1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        scheduler.step()
        train_epoch_loss = running_loss / len(train_loader)
        train_history.append(train_epoch_loss)

        model.eval()
        with torch.no_grad():
            test_preds = model(a_test_norm.to(device))
            test_loss = float(loss_fn(test_preds, u_test.to(device)).detach().cpu())
        test_history.append(test_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"epoch={epoch:4d} train={train_epoch_loss:.6f} test={test_loss:.6f}")

        if epoch % config.history_interval == 0 or epoch == config.epochs:
            np.savetxt(
                history_path,
                np.column_stack((np.arange(1, len(train_history) + 1), train_history, test_history)),
                delimiter=",",
                header="epoch,train_loss,test_loss",
                comments="",
            )

        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            save_training_state(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                {"train": train_history, "test": test_history},
                config,
                device,
            )

    sample_idx = max(0, min(config.sample_index, a_test.shape[0] - 1))
    with torch.no_grad():
        sample_pred = model(a_test_norm[sample_idx : sample_idx + 1].to(device)).squeeze(0).cpu()

    loss_plot_path = config.output_dir / "loss_curves.png"
    prediction_plot_path = config.output_dir / "prediction_comparison.png"
    model_path = config.output_dir / "model.pt"

    plot_loss_curves(train_history, test_history, loss_plot_path)
    plot_prediction_comparison(u_test[sample_idx], sample_pred, a_test[sample_idx], prediction_plot_path)
    np.savetxt(
        history_path,
        np.column_stack((np.arange(1, len(train_history) + 1), train_history, test_history)),
        delimiter=",",
        header="epoch,train_loss,test_loss",
        comments="",
    )
    torch.save({"model": model.state_dict(), "config": config.__dict__, "device": str(device)}, model_path)

    return {
        "loss_plot": loss_plot_path,
        "prediction_plot": prediction_plot_path,
        "history_csv": history_path,
        "model": model_path,
        "checkpoint": checkpoint_path,
    }
