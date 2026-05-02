"""
SmolLM2-135M with Extended Internal Token Vocabulary

在 SmolLM2-135M-Instruct 基础上扩展 embedding 层，支持 4096 个内部离散 token。
原始 token + 内部 token + 特殊 token = 53256。

RTX 3060 优化：
- BF16 mixed precision
- Gradient checkpointing
- torch.compile (可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional
import math
import warnings


class SmolLM2Internal(nn.Module):
    """
    扩展版 SmolLM2-135M，支持内部 token 词汇表。

    原始词表: 49152
    内部 token: 4096 (INTERNAL_0 ~ INTERNAL_4095)
    特殊 token: 8
    总计: 53256

    关键设计：
    - 加载预训练权重，扩展 embedding 和 lm_head
    - 新的 embedding 行用小随机值初始化
    - 支持 internal token mask（限制采样范围）
    """

    def __init__(
        self,
        base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        n_internal_tokens: int = 4096,
        n_special_tokens: int = 8,
        use_gradient_checkpointing: bool = True,
        device: str = "cuda",
        _skip_extend: bool = False,
    ):
        super().__init__()

        self.n_internal_tokens = n_internal_tokens
        self.n_special_tokens = n_special_tokens
        self.original_vocab_size = None  # set after loading
        self.total_vocab_size = None
        self.device = device

        # Load base model
        print(f"[SmolLM2Internal] Loading base model: {base_model}")
        self.config = AutoConfig.from_pretrained(base_model)
        self.original_vocab_size = self.config.vocab_size

        # 尝试使用 Flash Attention 2（大幅降低显存）
        attn_kwargs = {}
        try:
            import flash_attn  # noqa: F401
            attn_kwargs["attn_implementation"] = "flash_attention_2"
            print("[SmolLM2Internal] Flash Attention 2 enabled")
        except ImportError:
            # Fallback: 使用 SDPA（PyTorch 内置优化 attention，比 eager 省显存）
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                attn_kwargs["attn_implementation"] = "sdpa"
                print("[SmolLM2Internal] SDPA attention enabled (Flash Attn not found)")
            else:
                print("[SmolLM2Internal] WARNING: Using eager attention (high VRAM usage)")

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=None,  # manual placement
            low_cpu_mem_usage=True,
            **attn_kwargs,
        )

        # Extend vocabulary (skip when loading checkpoint — already extended)
        self.total_vocab_size = self.original_vocab_size + n_internal_tokens + n_special_tokens
        # Extend vocabulary (skip when loading checkpoint — already extended)
        if not _skip_extend:
            self.total_vocab_size = self.original_vocab_size + n_internal_tokens + n_special_tokens
            self._extend_embeddings()
        else:
            # Loading from checkpoint: embeddings already extended, use actual size
            embed = self.model.get_input_embeddings()
            self.original_vocab_size = self.original_vocab_size  # keep config value
            self.total_vocab_size = embed.weight.shape[0]
            print(f"[SmolLM2Internal] Skipped embedding extension (loading from checkpoint). "
                  f"Vocab: {self.total_vocab_size}")

        # Move to device
        self.model = self.model.to(device)

        # Gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("[SmolLM2Internal] Gradient checkpointing enabled")

        # Torch compile (optional, may fail on some CUDA versions)
        # Triton is required for inductor backend; unavailable on Windows.
        self._compiled = False
        try:
            import triton  # noqa: F401
            _triton_ok = True
        except ImportError:
            _triton_ok = False

        if _triton_ok and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                print("[SmolLM2Internal] torch.compile enabled")
            except Exception as e:
                print(f"[SmolLM2Internal] torch.compile skipped: {e}")
        else:
            print("[SmolLM2Internal] torch.compile skipped: Triton not available (Windows)")

        # Build internal token mask for logit restriction
        # 当 _skip_extend=True 时，original_vocab_size 从 checkpoint config 读取
        # 可能不正确（config.vocab_size 已被 resize 为 total），由 from_checkpoint
        # 在修正 original_vocab_size 后显式调用 _build_token_masks()
        if not _skip_extend:
            self._build_token_masks()
        # else: masks 由 from_checkpoint 构建

        print(f"[SmolLM2Internal] Initialized. Total vocab: {self.total_vocab_size}")

    def _extend_embeddings(self):
        """扩展 embedding 和 lm_head 以容纳内部 token。"""
        old_embed = self.model.get_input_embeddings()
        old_lm_head = self.model.get_output_embeddings()

        # Resize token embeddings
        self.model.resize_token_embeddings(self.total_vocab_size)

        new_embed = self.model.get_input_embeddings()

        # Initialize new embedding rows with small random values
        # (first original_vocab_size rows keep pretrained weights)
        with torch.no_grad():
            new_indices = slice(self.original_vocab_size, self.total_vocab_size)
            nn.init.normal_(
                new_embed.weight[new_indices],
                mean=0.0,
                std=0.02 / math.sqrt(self.config.hidden_size),
            )

        # Do the same for lm_head if it's tied (or separate)
        new_lm_head = self.model.get_output_embeddings()
        if new_lm_head is not new_embed:  # untied weights
            with torch.no_grad():
                nn.init.normal_(
                    new_lm_head.weight[new_indices],
                    mean=0.0,
                    std=0.02 / math.sqrt(self.config.hidden_size),
                )

        print(f"[SmolLM2Internal] Embedding extended: "
              f"{self.original_vocab_size} → {self.total_vocab_size}")

    def _build_token_masks(self):
        """构建 internal token 范围掩码。"""
        self.internal_start_id = self.original_vocab_size       # 49152
        self.internal_end_id = self.original_vocab_size + 1      # 49153
        self.internal_base_id = self.original_vocab_size + 2     # 49154
        self.internal_max_id = self.internal_base_id + self.n_internal_tokens - 1  # 53249

        self.nl_start_id = self.original_vocab_size + self.n_internal_tokens + 2     # 53250
        self.nl_end_id = self.original_vocab_size + self.n_internal_tokens + 3       # 53251
        self.pad_id = self.original_vocab_size + self.n_internal_tokens + 4           # 53252

        # Internal token mask: allows sampling only internal tokens + INTERNAL_END
        self.internal_token_mask = torch.zeros(self.total_vocab_size, dtype=torch.bool)
        self.internal_token_mask[self.internal_base_id:self.internal_max_id + 1] = True
        self.internal_token_mask[self.internal_end_id] = True

        # Natural language mask: allows sampling only original tokens + NL_END + PAD
        self.nl_token_mask = torch.zeros(self.total_vocab_size, dtype=torch.bool)
        self.nl_token_mask[:self.original_vocab_size] = True
        self.nl_token_mask[self.nl_end_id] = True
        self.nl_token_mask[self.pad_id] = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) — 用于 CE loss，-100 表示忽略
            loss_mask: (batch, seq_len) — 额外的 loss 掩码（True 表示计算 loss）

        Returns:
            dict with keys: loss, logits, perplexity
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )

        loss = outputs.loss
        logits = outputs.logits

        # Apply custom loss mask if provided
        if loss_mask is not None and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = loss_mask[..., 1:].contiguous()

            # Compute per-token CE loss
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.total_vocab_size),
                shift_labels.view(-1),
                reduction='none',
            )
            ce_loss = ce_loss.view(shift_labels.shape)

            # Apply mask and average
            masked_loss = (ce_loss * shift_mask.float()).sum() / shift_mask.float().sum().clamp(min=1)
            loss = masked_loss

        return {
            "loss": loss,
            "logits": logits,
        }

    @torch.no_grad()
    def generate_internal_tokens(
        self,
        prompt_ids: torch.Tensor,
        max_internal_tokens: int = 256,
        temperature: float = 1.2,
        top_p: float = 0.95,
        min_internal_tokens: int = 4,
        internal_region_start: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从 prompt 生成内部 token 序列。

        输入格式: <INTERNAL_START> [NL text] <INTERNAL_END>
        输出格式: ... <INTERNAL_END> <INT_x> <INT_y> ... <INTERNAL_END>

        采样约束: 在 interal 区域内只允许采样 internal token (mask 掉 NL token)

        Args:
            prompt_ids: (seq_len,) — prompt token IDs
            max_internal_tokens: 最大生成的内部 token 数
            temperature: 采样温度 (高 = 更多样)
            top_p: nucleus sampling 阈值
            min_internal_tokens: 最少生成的内部 token 数
            internal_region_start: 从哪个位置开始限制 internal token
                                   (默认从 prompt 最后一个位置开始)

        Returns:
            generated_ids: (total_len,) — 完整序列 (prompt + generated)
            internal_mask: (total_len,) — True 标记 internal token 位置
        """
        device = prompt_ids.device
        self.model.eval()

        internal_token_mask = self.internal_token_mask.to(device)
        internal_end_id = torch.tensor(self.internal_end_id, device=device)

        generated = prompt_ids.clone()
        internal_tokens_generated = 0
        internal_positions = []

        past_key_values = None
        current_input = prompt_ids.unsqueeze(0)  # (1, seq_len)

        for step in range(max_internal_tokens + 50):  # extra buffer
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]  # (vocab_size,)

            # Apply internal token restriction
            logits = logits.masked_fill(~internal_token_mask, float("-inf"))

            # Temperature scaling
            logits = logits / max(temperature, 0.01)

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() < self.internal_base_id:
                # Not an internal token (should be INTERNAL_END)
                if next_token.item() == self.internal_end_id:
                    generated = torch.cat([generated, next_token])
                    internal_positions.append(len(generated) - 1)
                    if internal_tokens_generated >= min_internal_tokens:
                        break
                # Skip non-internal
                continue

            generated = torch.cat([generated, next_token])
            internal_positions.append(len(generated) - 1)
            internal_tokens_generated += 1

            # Prepare next input
            current_input = next_token.unsqueeze(0)

            if internal_tokens_generated >= max_internal_tokens:
                break

        # Force INTERNAL_END if not present
        if generated[-1] != self.internal_end_id:
            generated = torch.cat([generated, internal_end_id.unsqueeze(0)])
            internal_positions.append(len(generated) - 1)

        # Build internal mask
        internal_mask = torch.zeros(len(generated), dtype=torch.bool)
        for pos in internal_positions:
            internal_mask[pos] = True

        return generated, internal_mask

    def get_trainable_params(self, train_all: bool = True):
        """返回可训练参数。"""
        if train_all:
            return self.parameters()
        else:
            # 只训练 embedding 扩展部分
            embed = self.model.get_input_embeddings()
            return [embed.weight[self.original_vocab_size:]]

    def save_checkpoint(self, path: str):
        """保存模型 checkpoint。"""
        self.model.save_pretrained(path)
        torch.save({
            "total_vocab_size": self.total_vocab_size,
            "original_vocab_size": self.original_vocab_size,
            "n_internal_tokens": self.n_internal_tokens,
        }, f"{path}/extended_config.pt")
        print(f"[SmolLM2Internal] Checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = "cuda",
        use_gradient_checkpointing: bool = True,
    ):
        """从 checkpoint 加载（embedding 已扩展，跳过 _extend_embeddings）。"""
        ext_cfg = torch.load(f"{path}/extended_config.pt", map_location="cpu")
        instance = cls(
            base_model=path,
            n_internal_tokens=ext_cfg["n_internal_tokens"],
            n_special_tokens=8,
            use_gradient_checkpointing=use_gradient_checkpointing,
            device=device,
            _skip_extend=True,
        )
        # 修正 original_vocab_size 并重建 token mask（__init__ 中用的是 config 的 53256）
        instance.original_vocab_size = ext_cfg["original_vocab_size"]
        instance._build_token_masks()
        return instance
