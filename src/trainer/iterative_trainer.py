"""
迭代式自蒸馏训练主控制器

协调整个多轮训练流程：
1. Round 0: 在自然语言数据上 fine-tune（基线）
2. Round 1+: 生成内部 token 序列 → 混合训练 → 评估 → 下一轮

核心循环:
  for k in 1..N:
    1. Rephrase: 用 Model_K 生成内部 token 序列
    2. Mix: 80% 内部数据 + 20% 原始数据
    3. Train: Model_{K+1}（从 Model_K 继承）
    4. Evaluate: 记录指标
    5. Save checkpoint
"""

import torch
import json
import os
import time
from typing import Optional, Dict, List
from pathlib import Path

from ..model.smollm2_internal import SmolLM2Internal
from ..data.dataset import (
    ChatDataset, InternalSequenceDataset,
    collate_chat_batch, collate_internal_batch,
    create_dataloaders,
)
from ..trainer.rephrase import RephraseGenerator
from ..trainer.training_loop import SingleRoundTrainer
from ..translator.model import InternalTranslator, train_translator_step
from ..eval.metrics import compute_metrics, print_metrics


class IterativeTrainer:
    """
    多轮迭代自蒸馏训练器。

    架构：
    - 主模型 (SmolLM2Internal, ~140M)
    - 翻译模型 (InternalTranslator, ~30M)
    - 复述生成器 (RephraseGenerator)
    """

    def __init__(
        self,
        config: Dict,
        tokenizer,  # ExtendedTokenizer
        device: str = "cuda",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Unpack config
        self.n_rounds = config.get("iterative", {}).get("n_rounds", 20)
        self.data_mix_ratio = config.get("iterative", {}).get("data_mix_ratio", 0.2)
        self.temperature_start = config.get("iterative", {}).get("sample_temperature_start", 1.2)
        self.temperature_end = config.get("iterative", {}).get("sample_temperature_end", 0.8)
        self.top_p = config.get("iterative", {}).get("sample_top_p", 0.95)
        self.max_internal_mult = config.get("iterative", {}).get("max_internal_tokens_multiplier", 3)

        self.batch_size = config.get("training", {}).get("batch_size", 8)
        self.grad_accum = config.get("training", {}).get("gradient_accumulation_steps", 2)
        self.lr = config.get("training", {}).get("learning_rate", 3e-4)
        self.max_steps_per_round = config.get("training", {}).get("max_steps_per_round", 5000)
        self.warmup_steps = config.get("training", {}).get("warmup_steps", 500)

        self.entropy_coeff = config.get("loss", {}).get("entropy_bonus_coeff", 0.05)
        self.alpha_trans = config.get("loss", {}).get("alpha_translation", 0.3)

        self.translator_cfg = config.get("translator", {})
        self.checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", "./checkpoints")
        self.save_every = config.get("logging", {}).get("save_every_n_steps", 1000)

        self.jsonl_path = config.get("data", {}).get("jsonl_path", "")
        self.max_samples = config.get("data", {}).get("max_samples", None)
        self.val_split = config.get("data", {}).get("val_split", 0.05)

        # Models (lazy init)
        self.main_model: Optional[SmolLM2Internal] = None
        self.translator: Optional[InternalTranslator] = None
        self.rephrase_gen: Optional[RephraseGenerator] = None
        self.current_round = 0

        # Results tracking
        self.results_log = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_temperature(self, round_num: int) -> float:
        """线性衰减采样温度。"""
        if self.n_rounds <= 1:
            return self.temperature_start
        progress = round_num / (self.n_rounds - 1)
        return self.temperature_start + (self.temperature_end - self.temperature_start) * progress

    def _estimate_internal_limit(self, nl_token_count: int) -> int:
        """估算最大内部 token 数。"""
        return min(nl_token_count * self.max_internal_mult, 256)

    def initialize_models(self):
        """初始化主模型和翻译模型。"""
        print("\n" + "=" * 60)
        print("  Initializing Models")
        print("=" * 60)

        # Main model
        self.main_model = SmolLM2Internal(
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            n_internal_tokens=4096,
            n_special_tokens=8,
            use_gradient_checkpointing=True,
            device=self.device,
        )

        # Translation model
        self.translator = InternalTranslator(
            src_vocab_size=self.main_model.total_vocab_size,
            tgt_vocab_size=self.main_model.original_vocab_size,
            d_model=self.translator_cfg.get("d_model", 384),
            n_heads=self.translator_cfg.get("n_heads", 6),
            n_encoder_layers=self.translator_cfg.get("n_encoder_layers", 6),
            n_decoder_layers=self.translator_cfg.get("n_decoder_layers", 6),
            d_ff=self.translator_cfg.get("d_ff", 1024),
        ).to(self.device)

        print(f"  Main model: {sum(p.numel() for p in self.main_model.parameters())/1e6:.1f}M params")
        print(f"  Translator: {self.translator.get_num_params()/1e6:.1f}M params")
        print(f"  Total: ~{(sum(p.numel() for p in self.main_model.parameters()) + self.translator.get_num_params())/1e6:.1f}M params")

    def run_round_0(self):
        """
        Round 0: 在自然语言数据上 fine-tune。

        这是基线训练，后续迭代的起点。
        """
        print("\n" + "=" * 60)
        print("  Round 0: Baseline Fine-tuning")
        print("=" * 60)

        # Data loaders
        train_loader, val_loader = create_dataloaders(
            jsonl_path=self.jsonl_path,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=2048,
            max_samples=self.max_samples,
            val_split=self.val_split,
        )

        # Trainer
        trainer = SingleRoundTrainer(
            model=self.main_model,
            tokenizer=self.tokenizer,
            device=self.device,
            learning_rate=self.lr,
            warmup_steps=self.warmup_steps,
            gradient_accumulation_steps=self.grad_accum,
            entropy_bonus_coeff=0.0,  # no entropy bonus for round 0
        )

        # Train
        metrics = trainer.train_round(
            train_loader=train_loader,
            val_loader=val_loader,
            max_steps=self.max_steps_per_round,
            round_num=0,
            use_internal_loss=False,
        )

        metrics["temperature"] = self.temperature_start
        self.results_log.append(metrics)

        # Save checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, "round_0")
        self.main_model.save_checkpoint(ckpt_path)
        self._save_translator(os.path.join(self.checkpoint_dir, "round_0", "translator.pt"))
        self._save_results()

        return metrics

    def run_iterative_round(self, round_num: int, train_samples: List[Dict]):
        """
        执行一轮迭代训练。

        Args:
            round_num: 当前轮次 (>= 1)
            train_samples: 训练样本列表 (从 ChatDataset)
        """
        print("\n" + "=" * 60)
        print(f"  Round {round_num}: Iterative Distillation")
        print("=" * 60)

        temp = self._get_temperature(round_num)
        print(f"  Temperature: {temp:.2f}")

        # ========== Phase 1: Rephrase ==========
        print(f"\n  [Phase 1] Generating internal token sequences...")
        self.rephrase_gen = RephraseGenerator(
            model=self.main_model,
            tokenizer=self.tokenizer,
            temperature=temp,
            top_p=self.top_p,
            max_internal_tokens=256,
            min_internal_tokens=4,
            device=self.device,
        )

        # Generate internal sequences for a subset of training data
        rephrase_samples = train_samples[:2000]  # 每轮处理 2000 条（可调）
        internal_sequences = self.rephrase_gen.generate_from_dataset(
            samples=rephrase_samples,
            text_key="text",
            show_progress=True,
        )

        print(f"  Generated {len(internal_sequences)} internal sequences")

        # ========== Phase 2: Data Mixing ==========
        print(f"\n  [Phase 2] Mixing data ({1-self.data_mix_ratio:.0%} internal + {self.data_mix_ratio:.0%} original)...")

        # Build internal token dataset
        internal_dataset = InternalSequenceDataset(internal_sequences)

        # Original data samples
        num_original = int(len(rephrase_samples) * self.data_mix_ratio)
        original_dataset = ChatDataset(
            jsonl_path=self.jsonl_path,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_samples=num_original,
            split="train",
            val_split=0.0,  # use all for original
        )

        # Create loaders
        pad_id = self.tokenizer.pad_id

        # Mix by alternating batches (simple approach)
        internal_loader = torch.utils.data.DataLoader(
            internal_dataset,
            batch_size=self.batch_size // 2,
            shuffle=True,
            collate_fn=lambda b: collate_internal_batch(b, pad_id),
            num_workers=0,
        )
        original_loader = torch.utils.data.DataLoader(
            original_dataset,
            batch_size=self.batch_size // 2,
            shuffle=True,
            collate_fn=lambda b: collate_chat_batch(b, pad_id),
            num_workers=0,
        )

        # Combine into interleaved loader
        mixed_loader = self._interleave_loaders(internal_loader, original_loader)

        # ========== Phase 3: Train ==========
        print(f"\n  [Phase 3] Training...")
        trainer = SingleRoundTrainer(
            model=self.main_model,
            tokenizer=self.tokenizer,
            device=self.device,
            learning_rate=self.lr * 0.5,  # lower LR for iterative rounds
            warmup_steps=min(200, self.warmup_steps // 2),
            gradient_accumulation_steps=self.grad_accum,
            entropy_bonus_coeff=self.entropy_coeff,
        )

        val_loader, _ = create_dataloaders(
            jsonl_path=self.jsonl_path,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=2048,
            max_samples=self.max_samples,
            val_split=self.val_split,
        )

        metrics = trainer.train_round(
            train_loader=mixed_loader,
            val_loader=val_loader,
            max_steps=self.max_steps_per_round // 2,  # fewer steps per round
            round_num=round_num,
            use_internal_loss=True,
        )

        # ========== Phase 4: Train Translator ==========
        print(f"\n  [Phase 4] Training translator...")
        self._train_translator(internal_sequences[:500], steps=500)

        # ========== Phase 5: Evaluate ==========
        print(f"\n  [Phase 5] Evaluation...")
        internal_token_seqs = [s["full_ids"] for s in internal_sequences]
        # Extract internal token IDs from full sequences
        internal_only = []
        nl_counts = []
        for seq_data in internal_sequences:
            mask = seq_data["internal_mask"]
            internal_part = seq_data["full_ids"][mask]
            nl_part = seq_data["full_ids"][~mask]
            internal_only.append(internal_part)
            nl_counts.append(len(nl_part))

        eval_metrics = compute_metrics(
            internal_sequences=internal_only,
            nl_token_counts=nl_counts,
            n_internal_tokens=4096,
            internal_base_id=self.main_model.internal_base_id,
        )
        print_metrics(eval_metrics, round_num)

        metrics.update(eval_metrics)
        metrics["temperature"] = temp
        self.results_log.append(metrics)

        # ========== Phase 6: Save ==========
        ckpt_path = os.path.join(self.checkpoint_dir, f"round_{round_num}")
        self.main_model.save_checkpoint(ckpt_path)
        self._save_translator(os.path.join(ckpt_path, "translator.pt"))
        self._save_results()

        return metrics

    def _interleave_loaders(self, loader_a, loader_b):
        """交替混合两个 data loader（80% internal + 20% original）。"""
        class InterleavedLoader:
            def __init__(self, a, b, ratio_a=0.8):
                self.a = a
                self.b = b
                self.a_iter = iter(a)
                self.b_iter = iter(b)
                self.ratio_a = ratio_a
                self.a_ratio = int(ratio_a * 10)
                self.b_ratio = 10 - self.a_ratio
                self.counter = 0

            def __iter__(self):
                return self

            def __next__(self):
                use_a = (self.counter % 10) < self.a_ratio
                self.counter += 1

                if use_a:
                    try:
                        return next(self.a_iter)
                    except StopIteration:
                        self.a_iter = iter(self.a)
                        return next(self.a_iter)
                else:
                    try:
                        return next(self.b_iter)
                    except StopIteration:
                        self.b_iter = iter(self.b)
                        return next(self.b_iter)

            def __len__(self):
                return min(len(self.a), len(self.b)) * 2

        return InterleavedLoader(loader_a, loader_b)

    def _train_translator(self, internal_sequences: List[Dict], steps: int = 500):
        """训练翻译模型几个 step。"""
        self.translator.train()
        optimizer = torch.optim.AdamW(self.translator.parameters(), lr=1e-4)

        total_loss = 0.0
        for step in range(steps):
            # Sample a batch from internal sequences
            batch_seqs = []
            batch_texts = []
            for _ in range(min(8, len(internal_sequences))):
                idx = torch.randint(0, len(internal_sequences), (1,)).item()
                seq_data = internal_sequences[idx]
                # Extract internal token part
                mask = seq_data["internal_mask"]
                internal_part = seq_data["full_ids"][mask]
                batch_seqs.append(internal_part)
                batch_texts.append(seq_data["nl_text"])

            # Pad internal sequences
            max_src_len = max(len(s) for s in batch_seqs)
            src_ids = torch.zeros(len(batch_seqs), max_src_len, dtype=torch.long)
            for i, s in enumerate(batch_seqs):
                src_ids[i, :len(s)] = s

            # Tokenize target texts
            tgt_sequences = []
            max_tgt_len = 0
            for text in batch_texts:
                tgt = self.tokenizer.encode(text, add_special_tokens=True,
                                             max_length=256, truncation=True)
                tgt_sequences.append(tgt)
                max_tgt_len = max(max_tgt_len, len(tgt))

            tgt_ids = torch.zeros(len(batch_seqs), max_tgt_len, dtype=torch.long)
            for i, t in enumerate(tgt_sequences):
                tgt_ids[i, :len(t)] = t

            src_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)

            loss = train_translator_step(
                self.translator, src_ids, tgt_ids, optimizer,
            )
            total_loss += loss

        avg_loss = total_loss / max(1, steps)
        print(f"  Translator avg loss: {avg_loss:.4f}")

    def run(self):
        """运行完整的多轮迭代训练。"""
        print("\n" + "=" * 60)
        print("  Iterative Token-Level Self-Distillation")
        print("  Multi-Round Language Evolution")
        print("=" * 60)
        print(f"  Rounds: {self.n_rounds}")
        print(f"  Data mix ratio: {self.data_mix_ratio}")
        print(f"  Temperature: {self.temperature_start} → {self.temperature_end}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Max steps/round: {self.max_steps_per_round}")
        print(f"  Device: {self.device}")
        print("=" * 60)

        # Initialize
        self.initialize_models()

        # Prepare training samples for rephrase
        train_ds = ChatDataset(
            jsonl_path=self.jsonl_path,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_samples=self.max_samples,
            split="train",
            val_split=self.val_split,
        )
        train_samples = train_ds._tokenized

        # Round 0: Baseline
        print(f"\n{'#'*60}")
        print(f"  PHASE 1: Round 0 — Baseline Training")
        print(f"{'#'*60}")
        r0_metrics = self.run_round_0()
        print(f"\n  Round 0 complete. Val loss: {r0_metrics.get('val_loss', 'N/A')}")

        # Iterative rounds
        for r in range(1, self.n_rounds + 1):
            print(f"\n{'#'*60}")
            print(f"  PHASE 2: Round {r}/{self.n_rounds} — Iterative Distillation")
            print(f"{'#'*60}")

            try:
                metrics = self.run_iterative_round(r, train_samples)
                print(f"\n  Round {r} complete.")
                print(f"  Entropy: {metrics.get('entropy', 'N/A'):.3f}")
                print(f"  Token usage: {metrics.get('effective_token_usage', {}).get('usage_rate', 'N/A')}")

            except KeyboardInterrupt:
                print(f"\n  Interrupted at round {r}. Saving checkpoint...")
                ckpt_path = os.path.join(self.checkpoint_dir, f"round_{r}_interrupted")
                self.main_model.save_checkpoint(ckpt_path)
                self._save_results()
                break
            except Exception as e:
                print(f"\n  Error in round {r}: {e}")
                import traceback
                traceback.print_exc()
                self._save_results()
                break

        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        self._save_results()
        self._print_summary()

    def _save_translator(self, path: str):
        """保存翻译模型。"""
        torch.save(self.translator.state_dict(), path)

    def _save_results(self):
        """保存结果日志。"""
        path = os.path.join(self.checkpoint_dir, "results.json")
        with open(path, "w") as f:
            json.dump(self.results_log, f, indent=2, default=str)

    def _print_summary(self):
        """打印训练摘要。"""
        print("\n  Training Summary:")
        print("  " + "-" * 40)
        for r in self.results_log:
            rn = r.get("round", "?")
            tl = r.get("train_loss", "N/A")
            vl = r.get("val_loss", "N/A")
            ent = r.get("entropy", "N/A")
            print(f"  Round {rn}: train_loss={tl}, val_loss={vl}, entropy={ent}")
