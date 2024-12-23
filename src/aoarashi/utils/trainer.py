import json
import math
import os
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

logger = getLogger(__name__)


class Trainer:
    def __init__(
        self, model: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, config: DictConfig
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(self.config.out_dir)
        # load checkpoint
        if os.path.exists(self.config.checkpoint_path):
            self._load_checkpoint()
        else:
            self.best_valid_loss = float("inf")
            self.start_epoch = 0
        # write train config
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.out_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(OmegaConf.to_container(config, resolve=True), f, ensure_ascii=False, indent=4)

    def _get_optimizer(self):
        if self.config.optimizer.type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                eps=self.config.optimizer.eps,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif self.config.optimizer.type == "adamw":
            return optim.AdamW(
                self.model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay
            )
        else:
            raise ValueError(f"invalid optimizer name: {self.config.optimizer.name}")

    def _get_scheduler(self):
        total_step = len(self.train_dataloader) // self.config.grad_accum_steps * self.config.epochs
        warmup_steps = self.config.scheduler.warmup_steps // self.config.grad_accum_steps
        if self.config.scheduler.type == "linear":
            return LambdaLR(
                self.optimizer,
                lambda s: s / warmup_steps if s < warmup_steps else (total_step - s) / (total_step - warmup_steps),
            )
        elif self.config.scheduler.type == "transformer":
            # NOTE: avoid division where step starts from 1
            return LambdaLR(
                self.optimizer,
                lambda s: 0.0 if s == 0 else min(s**-0.5, s * warmup_steps**-1.5) * math.sqrt(warmup_steps),
            )
        else:
            raise ValueError(f"invalid scheduler type: {self.config.scheduler.type}")

    def _train_epoch(self, epoch: int):
        logger.info(f"epoch {epoch + 1} training")
        self.model.train()
        self.optimizer.zero_grad()
        epoch_loss = 0.0
        for step, batch in enumerate(self.train_dataloader, 1):
            with autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                loss, stats = self.model(**{k: v.to(self.device) for k, v in batch.items()})
                loss = loss / self.config.grad_accum_steps
            self.scaler.scale(loss).backward()
            if step % self.config.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            if step % self.config.log_steps == 0:
                msg = (
                    f"epoch: {epoch + 1}/{self.config.epochs}, "
                    f"step: {step:,}/{len(self.train_dataloader):,}, "
                    f"lr: {self.scheduler.get_last_lr()[0]:.6f}, "
                )
                for k, v in stats.items():
                    msg += f"{k}: {v:.3f}, "
                logger.info(msg)
            epoch_loss += stats["loss"]
        return epoch_loss / len(self.train_dataloader)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int):
        logger.info(f"epoch {epoch + 1} validation")
        self.model.eval()
        epoch_loss = 0.0
        for batch in self.valid_dataloader:
            _, stats = self.model(**{k: v.to(self.device) for k, v in batch.items()})
            epoch_loss += stats["loss"]
        return epoch_loss / len(self.valid_dataloader)

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_valid_loss": self.best_valid_loss,
        }
        torch.save(state, self.out_dir / f"checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(self.model.state_dict(), self.out_dir / "best_model.pt")

    def _load_checkpoint(self):
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_valid_loss = checkpoint["best_valid_loss"]
        self.start_epoch = checkpoint["epoch"]

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._train_epoch(epoch)
            valid_loss = self._validate_epoch(epoch)
            logger.info(
                f"epoch {epoch + 1}/{self.config.epochs}, "
                f"train loss: {train_loss:.3f}, "
                f"valid loss: {valid_loss:.3f}"
            )
            is_best = valid_loss < self.best_valid_loss
            if is_best:
                self.best_valid_loss = valid_loss
            self._save_checkpoint(epoch + 1, is_best)
