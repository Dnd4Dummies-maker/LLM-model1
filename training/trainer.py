import math
import time

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 500,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.device = device
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler(device) if self.use_amp else None

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            ctx = torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16) if self.use_amp else torch.autocast(self.device, enabled=False)
            with ctx:
                _, loss = self.model(input_ids, targets)
                loss = loss / self.grad_accum_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps

            if (step + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            if (step + 1) % self.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                elapsed = time.time() - start_time
                tokens_per_sec = (self.log_interval * self.train_loader.batch_size * input_ids.shape[1]) / elapsed
                print(
                    f"Epoch {epoch} | Step {step + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Tokens/s: {tokens_per_sec:.0f}"
                )
                start_time = time.time()

            if (step + 1) % self.eval_interval == 0 and self.val_loader:
                self.evaluate()

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                ctx = torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16) if self.use_amp else torch.autocast(self.device, enabled=False)
                with ctx:
                    _, loss = self.model(input_ids, targets)
                total_loss += loss.item()
                n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(avg_loss)
        print(f"  Validation | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        self.model.train()

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            self.train_epoch(epoch)
            if self.val_loader:
                self.evaluate()
