from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .checkpoints import load_model_weights
from .config import ExperimentConfig
from .datasets import KOREATECHHolographyDataset
from .encoders import KOREATECHCGHEncoder
from .models import build_model


def create_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    encoder = KOREATECHCGHEncoder(config.encoder)
    train_dataset = KOREATECHHolographyDataset(config.data_root / "train", encoder)
    val_dataset = KOREATECHHolographyDataset(config.data_root / "validation", encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    return train_loader, val_loader


def _build_writer(config: ExperimentConfig) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = config.log_dir / f"{config.experiment_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))


def run_training(config: ExperimentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders(config)

    model = build_model(config.model, config.encoder).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    writer = _build_writer(config)
    start_epoch = 0

    if config.train.resume_checkpoint:
        checkpoint_path = config.resolve_path(config.train.resume_checkpoint)
        if checkpoint_path.exists():
            start_epoch = load_model_weights(model, checkpoint_path, device)

    config.weight_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("batch/loss", loss.item(), global_step)

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("epoch/train_loss", epoch_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch)
        scheduler.step(epoch_loss)

        with torch.no_grad():
            vis_amp = outputs[0, 0:1].detach().cpu()
            vis_phs = outputs[0, 1:2].detach().cpu()
            writer.add_image("visuals/predicted_amplitude", vis_amp, epoch)
            writer.add_image("visuals/predicted_phase", vis_phs, epoch)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{config.train.epochs} "
            f"- train_loss={epoch_loss:.6f} "
            f"- val_loss={avg_val_loss:.6f} "
            f"- time={elapsed:.2f}s"
        )

        if (epoch + 1) % config.train.checkpoint_every == 0:
            checkpoint_name = f"{config.experiment_name}_epoch_{epoch + 1}.pth"
            checkpoint_path = config.weight_dir / checkpoint_name
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    writer.close()


def inspect_dataset(config: ExperimentConfig) -> tuple[torch.Size, torch.Size]:
    train_loader, _ = create_dataloaders(config)
    sample_inputs, sample_targets = next(iter(train_loader))
    return sample_inputs.shape, sample_targets.shape
