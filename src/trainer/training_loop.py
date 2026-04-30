"""
单轮训练循环

支持：
- 标准 LM 训练（Round 0）
- 内部 token 序列训练（Round 1+）
- BF16 mixed precision
- 梯度累积
- Entropy bonus（防坍塌）
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Callable
import time
import math
from tqdm import tqdm


class SingleRoundTrainer:
    """
    单轮训练器。在一个 epoch 内训练模型。

    Round 0: 标准 LM 训练（自然语言数据）
    Round 1+: 内部 token 序列训练 + 20% 原始数据混合
    """

    def __init__(
        self,
        model,  # SmolLM2Internal
        tokenizer,  # ExtendedTokenizer
        device: str = "cuda",
        learning_rate: float = 3e-4,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 2,
        entropy_bonus_coeff: float = 0.05,
        use_bf16: bool = True,
        log_every: int = 50,
        eval_every: int = 500,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.use_bf16 = use_bf16
        self.log_every = log_every
        self.eval_every = eval_every
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # LR scheduler (cosine with warmup)
        self.warmup_steps = warmup_steps
        self.lr = learning_rate
        self._total_steps = 0
        self._scheduler = None

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Cosine decay with linear warmup。"""
        if step < self.warmup_steps:
            return self.lr * (step / max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _compute_entropy_bonus(self, logits: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        计算熵奖励（鼓励 token 多样性）。

        H = -Σ p(x) log p(x)
        Bonus = -entropy_bonus_coeff * mean(H per position)

        高熵 → bonus 更负 → loss 更低 → 鼓励多样性
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_per_position = -(probs * log_probs).sum(dim=-1)  # (batch, seq)

        # 只在 loss_mask 位置计算熵
        if loss_mask.any():
            masked_entropy = entropy_per_position[loss_mask].mean()
        else:
            masked_entropy = entropy_per_position.mean()

        return -self.entropy_bonus_coeff * masked_entropy

    def train_round(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        max_steps: int = 5000,
        round_num: int = 0,
        use_internal_loss: bool = False,
        callback: Optional[Callable] = None,
    ) -> dict:
        """
        执行一轮训练。

        Args:
            train_loader: 训练数据
            val_loader: 验证数据（可选）
            max_steps: 最大训练步数
            round_num: 当前轮次
            use_internal_loss: 是否使用内部 token loss mask
            callback: 每 N 步回调 (step, metrics) -> None

        Returns:
            dict with training metrics
        """
        self.model.model.train()
        total_steps = max_steps

        train_losses = []
        val_losses = []
        entropies = []
        start_time = time.time()
        self.optimizer.zero_grad()

        # Create scheduler
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: self._get_lr(s, total_steps) / self.lr
        )

        data_iter = iter(train_loader)
        global_step = 0
        pbar = tqdm(total=max_steps, desc=f"Round {round_num}", unit="step")

        while global_step < max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_bf16 else torch.no_grad():
                if use_internal_loss and "loss_mask" in batch:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        loss_mask=batch["loss_mask"],
                    )
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                loss = outputs["loss"]

                # Entropy bonus (只在内部 token 区域)
                if use_internal_loss and "loss_mask" in batch:
                    entropy_bonus = self._compute_entropy_bonus(
                        outputs["logits"], batch["loss_mask"]
                    )
                    loss = loss + entropy_bonus

                # Scale for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward
            loss.backward()

            global_step += 1
            accum_step = global_step % self.gradient_accumulation_steps

            if accum_step == 0 or global_step >= max_steps:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.optimizer.step()
                self._scheduler.step()
                self.optimizer.zero_grad()

            # Logging
            step_loss = loss.item() * self.gradient_accumulation_steps
            train_losses.append(step_loss)

            pbar.set_postfix({
                "loss": f"{step_loss:.4f}",
                "lr": f"{self._scheduler.get_last_lr()[0]:.2e}",
            })
            pbar.update(1)

            if global_step % self.log_every == 0:
                avg_loss = sum(train_losses[-100:]) / len(train_losses[-100:])
                print(f"  Step {global_step}/{max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {self._scheduler.get_last_lr()[0]:.2e}")

            # Validation
            if val_loader is not None and global_step % self.eval_every == 0:
                val_loss = self._evaluate(val_loader)
                val_losses.append((global_step, val_loss))
                self.model.model.train()

            # Callback
            if callback is not None and global_step % self.log_every == 0:
                callback(global_step, {
                    "train_loss": step_loss,
                    "round": round_num,
                })

        pbar.close()

        elapsed = time.time() - start_time
        final_val_loss = self._evaluate(val_loader) if val_loader is not None else None

        metrics = {
            "round": round_num,
            "steps": global_step,
            "train_loss": sum(train_losses[-100:]) / len(train_losses[-100:]),
            "val_loss": final_val_loss,
            "elapsed_hours": elapsed / 3600,
        }

        print(f"\n[Round {round_num}] Complete. "
              f"Train loss: {metrics['train_loss']:.4f}, "
              f"Val loss: {final_val_loss:.4f if final_val_loss else 'N/A'}, "
              f"Time: {elapsed/3600:.2f}h")

        return metrics

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader, max_batches: int = 50) -> float:
        """评估验证集 loss。"""
        self.model.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_bf16 else torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"]

            # Weight by number of non-ignored labels
            n_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        return total_loss / max(1, total_tokens)
