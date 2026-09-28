from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


class DenseNet(nn.Module):
    def __init__(self, layers: list[int], activation: type[nn.Module]) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                modules.append(activation())
        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class PinnConfig:
    data_path: Path
    output_dir: Path
    iterations: int = 5_000
    learning_rate: float = 1e-3
    step_size: int = 2_000
    gamma: float = 0.5
    log_interval: int = 100
    seed: int = 7
    measurement_weight: float = 100.0
    use_measurements: bool = False
    measurement_count: int = 50
    disp_layers: tuple[int, ...] = (2, 128, 128, 128, 2)
    stress_layers: tuple[int, ...] = (2, 128, 128, 128, 3)
    tau_right: float = 0.1
    tau_top: float = 0.0
    youngs_modulus: float = 10.0
    poisson_ratio: float = 0.3
    checkpoint_interval: int = 500
    history_interval: int = 100
    resume_from: Path | None = None


@dataclass
class PlateData:
    left_boundary: torch.Tensor
    right_boundary: torch.Tensor
    top_boundary: torch.Tensor
    bottom_boundary: torch.Tensor
    circle_boundary: torch.Tensor
    all_boundary: torch.Tensor
    displacement_truth: torch.Tensor
    triangulation: np.ndarray
    interior_points: torch.Tensor
    full_points: torch.Tensor

    def to(self, device: torch.device) -> PlateData:
        return PlateData(
            left_boundary=self.left_boundary.to(device),
            right_boundary=self.right_boundary.to(device),
            top_boundary=self.top_boundary.to(device),
            bottom_boundary=self.bottom_boundary.to(device),
            circle_boundary=self.circle_boundary.to(device),
            all_boundary=self.all_boundary.to(device),
            displacement_truth=self.displacement_truth.to(device),
            triangulation=self.triangulation,
            interior_points=self.interior_points.to(device),
            full_points=self.full_points.to(device),
        )


def _to_tensor(array: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float64, requires_grad=requires_grad)


def load_plate_data(path: Path) -> PlateData:
    if not path.exists():
        raise FileNotFoundError(f"Missing plate data file at {path}. Run plate_fem/Plate_hole.m first.")

    raw = scipy.io.loadmat(path)
    triangulation = np.asarray(raw["t"], dtype=np.int64) - 1
    return PlateData(
        left_boundary=_to_tensor(raw["L_boundary"]),
        right_boundary=_to_tensor(raw["R_boundary"]),
        top_boundary=_to_tensor(raw["T_boundary"]),
        bottom_boundary=_to_tensor(raw["B_boundary"]),
        circle_boundary=_to_tensor(raw["C_boundary"]),
        all_boundary=_to_tensor(raw["Boundary"], requires_grad=True),
        displacement_truth=_to_tensor(raw["disp_data"]),
        triangulation=triangulation,
        interior_points=_to_tensor(raw["p"], requires_grad=True),
        full_points=_to_tensor(raw["p_full"], requires_grad=True),
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plane_stress_stiffness(youngs_modulus: float, poisson_ratio: float) -> torch.Tensor:
    coeff = youngs_modulus / (1.0 - poisson_ratio**2)
    return coeff * torch.tensor(
        [
            [1.0, poisson_ratio, 0.0],
            [poisson_ratio, 1.0, 0.0],
            [0.0, 0.0, (1.0 - poisson_ratio) / 2.0],
        ],
        dtype=torch.float64,
    )


def compute_strain(disp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    u = disp[:, 0]
    v = disp[:, 1]
    dudx = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dvdx = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    e11 = dudx[:, 0:1]
    e22 = dvdx[:, 1:2]
    e12 = 0.5 * (dudx[:, 1:2] + dvdx[:, 0:1])
    return torch.cat((e11, e22, e12), dim=1)


def compute_constitutive_stress(disp: torch.Tensor, coords: torch.Tensor, stiffness: torch.Tensor) -> torch.Tensor:
    strain = compute_strain(disp, coords).unsqueeze(-1)
    expanded_stiffness = stiffness.unsqueeze(0).expand(strain.shape[0], -1, -1)
    return torch.bmm(expanded_stiffness, strain).squeeze(-1)


def select_measurements(
    points: torch.Tensor, displacement_truth: torch.Tensor, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(points.shape[0], generator=generator)[:count]
    return points[idx].detach().clone(), displacement_truth[idx].detach().clone()


def build_models(config: PinnConfig) -> tuple[DenseNet, DenseNet]:
    disp_net = DenseNet(list(config.disp_layers), nn.Tanh).double()
    stress_net = DenseNet(list(config.stress_layers), nn.Tanh).double()
    return disp_net, stress_net


def compute_losses(
    disp_net: DenseNet,
    stress_net: DenseNet,
    data: PlateData,
    stiffness: torch.Tensor,
    loss_fn: nn.Module,
    config: PinnConfig,
    measurement_points: torch.Tensor | None = None,
    measurement_disp: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    sigma = stress_net(data.interior_points)
    interior_disp = disp_net(data.interior_points)
    sigma_aug = compute_constitutive_stress(interior_disp, data.interior_points, stiffness)
    constitutive_loss = loss_fn(sigma_aug, sigma)

    boundary_disp = disp_net(data.all_boundary)
    boundary_sigma = stress_net(data.all_boundary)
    boundary_sigma_aug = compute_constitutive_stress(boundary_disp, data.all_boundary, stiffness)
    constitutive_boundary_loss = loss_fn(boundary_sigma_aug, boundary_sigma)

    sig11 = sigma[:, 0]
    sig22 = sigma[:, 1]
    sig12 = sigma[:, 2]

    dsig11 = torch.autograd.grad(sig11, data.interior_points, grad_outputs=torch.ones_like(sig11), create_graph=True)[0]
    dsig22 = torch.autograd.grad(sig22, data.interior_points, grad_outputs=torch.ones_like(sig22), create_graph=True)[0]
    dsig12 = torch.autograd.grad(sig12, data.interior_points, grad_outputs=torch.ones_like(sig12), create_graph=True)[0]

    equilibrium_x = dsig11[:, 0] + dsig12[:, 1]
    equilibrium_y = dsig12[:, 0] + dsig22[:, 1]
    equilibrium_loss_x = loss_fn(equilibrium_x, torch.zeros_like(equilibrium_x))
    equilibrium_loss_y = loss_fn(equilibrium_y, torch.zeros_like(equilibrium_y))

    left_disp = disp_net(data.left_boundary)
    bottom_disp = disp_net(data.bottom_boundary)
    right_sigma = stress_net(data.right_boundary)
    top_sigma = stress_net(data.top_boundary)
    circle_sigma = stress_net(data.circle_boundary)

    bc_left = loss_fn(left_disp[:, 0], torch.zeros_like(left_disp[:, 0]))
    bc_bottom = loss_fn(bottom_disp[:, 1], torch.zeros_like(bottom_disp[:, 1]))
    bc_right = loss_fn(right_sigma[:, 0], config.tau_right * torch.ones_like(right_sigma[:, 0])) + loss_fn(
        right_sigma[:, 2], torch.zeros_like(right_sigma[:, 2])
    )
    bc_top = loss_fn(top_sigma[:, 1], config.tau_top * torch.ones_like(top_sigma[:, 1])) + loss_fn(
        top_sigma[:, 2], torch.zeros_like(top_sigma[:, 2])
    )

    normals = data.circle_boundary
    circle_tx = circle_sigma[:, 0] * normals[:, 0] + circle_sigma[:, 2] * normals[:, 1]
    circle_ty = circle_sigma[:, 2] * normals[:, 0] + circle_sigma[:, 1] * normals[:, 1]
    bc_circle = loss_fn(circle_tx, torch.zeros_like(circle_tx)) + loss_fn(circle_ty, torch.zeros_like(circle_ty))

    total = (
        constitutive_loss
        + constitutive_boundary_loss
        + equilibrium_loss_x
        + equilibrium_loss_y
        + bc_left
        + bc_bottom
        + bc_right
        + bc_top
        + bc_circle
    )

    losses: dict[str, torch.Tensor] = {
        "total": total,
        "equilibrium_x": equilibrium_loss_x,
        "equilibrium_y": equilibrium_loss_y,
        "constitutive": constitutive_loss,
        "constitutive_boundary": constitutive_boundary_loss,
        "bc_left": bc_left,
        "bc_bottom": bc_bottom,
        "bc_right": bc_right,
        "bc_top": bc_top,
        "bc_circle": bc_circle,
    }

    if config.use_measurements and measurement_points is not None and measurement_disp is not None:
        measured_pred = disp_net(measurement_points)
        measurement_loss = loss_fn(measured_pred, measurement_disp)
        losses["measurement"] = measurement_loss
        losses["total"] = total + config.measurement_weight * measurement_loss

    return losses


def plot_loss_history(history: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["loss"], label="Training loss")
    if history.get("measurement"):
        plt.plot(history["epoch"], history["measurement"], label="Measurement loss")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sigma_field(
    full_points: torch.Tensor, triangulation: np.ndarray, sigma11: np.ndarray, output_path: Path
) -> None:
    triang = mtri.Triangulation(
        full_points[:, 0].detach().cpu().numpy(),
        full_points[:, 1].detach().cpu().numpy(),
        triangulation,
    )
    plt.figure(figsize=(7, 6))
    contour = plt.tricontourf(triang, sigma11, levels=40, cmap="viridis")
    plt.colorbar(contour, label=r"$\sigma_{11}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Predicted tensile stress field $\sigma_{11}$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_displacement_comparison(
    full_points: torch.Tensor,
    triangulation: np.ndarray,
    pred_disp: torch.Tensor,
    true_disp: torch.Tensor,
    output_path: Path,
) -> None:
    triang = mtri.Triangulation(
        full_points[:, 0].detach().cpu().numpy(),
        full_points[:, 1].detach().cpu().numpy(),
        triangulation,
    )
    pred = pred_disp[:, 0].detach().cpu().numpy()
    truth = true_disp[:, 0].detach().cpu().numpy()
    err = np.abs(pred - truth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        (truth, pred, err),
        ("FEM $u_x$", "PINN $u_x$", "|error|"),
        strict=True,
    ):
        contour = ax.tricontourf(triang, field, levels=40, cmap="viridis")
        fig.colorbar(contour, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_history(history: dict[str, list[float]], output_path: Path) -> None:
    measurement_history = history.get("measurement", [])
    padded_measurement = measurement_history + [math.nan] * (len(history["epoch"]) - len(measurement_history))
    stacked = np.column_stack(
        [
            np.asarray(history["epoch"], dtype=np.int64),
            np.asarray(history["loss"], dtype=np.float64),
            np.asarray(padded_measurement, dtype=np.float64),
        ]
    )
    np.savetxt(output_path, stacked, delimiter=",", header="iteration,total_loss,measurement_loss", comments="")


def save_training_state(
    output_path: Path,
    epoch: int,
    disp_net: DenseNet,
    stress_net: DenseNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    history: dict[str, list[float]],
    config: PinnConfig,
    device: torch.device,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "disp_net": disp_net.state_dict(),
            "stress_net": stress_net.state_dict(),
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
    disp_net: DenseNet,
    stress_net: DenseNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, dict[str, list[float]]]:
    state = torch.load(checkpoint_path, map_location=device)
    disp_net.load_state_dict(state["disp_net"])
    stress_net.load_state_dict(state["stress_net"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    history = state.get("history", {"epoch": [], "loss": [], "measurement": []})
    return int(state.get("epoch", 0)), history


def train_pinn(config: PinnConfig) -> dict[str, Path]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    print(f"Using device: {device}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_plate_data(config.data_path).to(device)
    disp_net, stress_net = build_models(config)
    disp_net = disp_net.to(device)
    stress_net = stress_net.to(device)
    stiffness = plane_stress_stiffness(config.youngs_modulus, config.poisson_ratio).to(device)
    loss_fn = nn.MSELoss()
    params = list(disp_net.parameters()) + list(stress_net.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    measurement_points = None
    measurement_disp = None
    if config.use_measurements:
        measurement_points, measurement_disp = select_measurements(
            data.full_points, data.displacement_truth, config.measurement_count, config.seed
        )
        measurement_points = measurement_points.to(device)
        measurement_disp = measurement_disp.to(device)

    history: dict[str, list[float]] = {"epoch": [], "loss": [], "measurement": []}
    start_epoch = 0
    checkpoint_path = config.output_dir / "checkpoint_latest.pt"
    history_path = config.output_dir / "training_history.csv"
    if config.resume_from is not None:
        start_epoch, history = load_training_state(
            config.resume_from, disp_net, stress_net, optimizer, scheduler, device
        )
        print(f"Resuming from checkpoint: {config.resume_from} at iteration {start_epoch}")

    for epoch in range(start_epoch + 1, config.iterations + 1):
        optimizer.zero_grad()
        losses = compute_losses(
            disp_net,
            stress_net,
            data,
            stiffness,
            loss_fn,
            config,
            measurement_points=measurement_points,
            measurement_disp=measurement_disp,
        )
        losses["total"].backward()
        optimizer.step()
        scheduler.step()

        history["epoch"].append(epoch)
        history["loss"].append(float(losses["total"].detach().cpu()))
        if "measurement" in losses:
            history["measurement"].append(float(losses["measurement"].detach().cpu()))

        if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.iterations:
            bc_total = (
                losses["bc_left"] + losses["bc_bottom"] + losses["bc_right"] + losses["bc_top"] + losses["bc_circle"]
            ).detach()
            msg = (
                f"iter={epoch:6d} total={history['loss'][-1]:.6e} "
                f"eq=({float(losses['equilibrium_x'].detach()):.3e}, {float(losses['equilibrium_y'].detach()):.3e}) "
                f"bc={float(bc_total):.3e}"
            )
            if "measurement" in losses:
                msg += f" meas={float(losses['measurement'].detach()):.3e}"
            print(msg)

        if epoch % config.history_interval == 0 or epoch == config.iterations:
            save_history(history, history_path)

        if epoch % config.checkpoint_interval == 0 or epoch == config.iterations:
            save_training_state(
                checkpoint_path,
                epoch,
                disp_net,
                stress_net,
                optimizer,
                scheduler,
                history,
                config,
                device,
            )

    with torch.enable_grad():
        full_disp = disp_net(data.full_points)
        full_sigma = compute_constitutive_stress(full_disp, data.full_points, stiffness)

    loss_plot_path = config.output_dir / "training_loss.png"
    sigma_plot_path = config.output_dir / "sigma11_contour.png"
    disp_plot_path = config.output_dir / "ux_comparison.png"
    model_path = config.output_dir / "models.pt"

    plot_loss_history(history, loss_plot_path)
    plot_sigma_field(data.full_points, data.triangulation, full_sigma[:, 0].detach().cpu().numpy(), sigma_plot_path)
    plot_displacement_comparison(
        data.full_points, data.triangulation, full_disp.detach(), data.displacement_truth, disp_plot_path
    )
    save_history(history, history_path)
    torch.save(
        {
            "disp_net": disp_net.state_dict(),
            "stress_net": stress_net.state_dict(),
            "config": config.__dict__,
            "device": str(device),
        },
        model_path,
    )

    return {
        "loss_plot": loss_plot_path,
        "sigma_plot": sigma_plot_path,
        "displacement_plot": disp_plot_path,
        "history_csv": history_path,
        "model": model_path,
        "checkpoint": checkpoint_path,
    }
