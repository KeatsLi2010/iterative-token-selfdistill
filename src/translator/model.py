"""
翻译模型：内部语言 → 自然语言

架构：小型 Encoder-Decoder Transformer
- Encoder: 处理内部 token 序列
- Decoder: 生成自然语言 token

训练方式：回译 + 循环一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import contextlib
from typing import Optional


class PositionalEncoding(nn.Module):
    """正弦位置编码。"""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(1), :])


class InternalTranslator(nn.Module):
    """
    内部语言 → 自然语言翻译模型。

    Encoder-Decoder Transformer。
    - Encoder 输入: 内部 token 序列 (4096 tokens)
    - Decoder 输出: 自然语言 (49152 tokens)
    """

    def __init__(
        self,
        src_vocab_size: int = 53256,       # 总词表（含内部 token）
        tgt_vocab_size: int = 49152,       # 自然语言词表
        d_model: int = 384,
        n_heads: int = 6,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Initialize
        self._init_weights()

        # Count params
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[InternalTranslator] Initialized. "
              f"Params: {n_params/1e6:.1f}M, "
              f"d_model={d_model}, n_heads={n_heads}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass。

        Args:
            src_ids: (batch, src_len) — 内部 token 序列
            tgt_ids: (batch, tgt_len) — 目标自然语言序列
            src_mask: 可选，encoder mask
            tgt_mask: 因果 mask for decoder
            src_padding_mask: (batch, src_len) — True 表示 padding
            tgt_padding_mask: (batch, tgt_len)

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # Embed
        src_emb = self.src_embed(src_ids) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.tgt_embed(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        # Encoder
        memory = self.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )

        # Decoder (with causal mask)
        if tgt_mask is None:
            tgt_len = tgt_ids.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt_ids.device),
                diagonal=1,
            )

        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        logits = self.output_proj(output)
        return logits

    def compute_loss(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算翻译损失（CE loss）。

        Args:
            src_ids: (batch, src_len)
            tgt_ids: (batch, tgt_len) — 目标序列（含 BOS 和 EOS）

        Returns:
            scalar loss
        """
        # Shift for teacher forcing
        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:].clone()
        if tgt_padding_mask is not None:
            tgt_output = tgt_output.masked_fill(tgt_padding_mask[:, 1:], -100)

        logits = self.forward(
            src_ids=src_ids,
            tgt_ids=tgt_input,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask[:, :-1] if tgt_padding_mask is not None else None,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, self.tgt_vocab_size),
            tgt_output.reshape(-1),
            ignore_index=-100,
            reduction='mean',
        )

        return loss

    @torch.no_grad()
    def translate(
        self,
        src_ids: torch.Tensor,
        tokenizer,  # ExtendedTokenizer for decoding
        max_len: int = 512,
        temperature: float = 0.8,
    ) -> str:
        """
        将内部 token 序列翻译为自然语言。

        Args:
            src_ids: (src_len,) — 内部 token 序列
            tokenizer: ExtendedTokenizer
            max_len: 最大生成长度
            temperature: 采样温度

        Returns:
            翻译后的自然语言文本
        """
        self.eval()
        device = next(self.parameters()).device
        src_ids = src_ids.unsqueeze(0).to(device)  # (1, src_len)

        # Encode
        src_emb = self.src_embed(src_ids) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)

        # Greedy decode
        bos_id = tokenizer.base_tokenizer.bos_token_id or 0
        generated = [bos_id]

        for _ in range(max_len):
            tgt_ids = torch.tensor([generated], device=device)
            tgt_emb = self.tgt_embed(tgt_ids) * math.sqrt(self.d_model)
            tgt_emb = self.pos_decoder(tgt_emb)

            tgt_mask = torch.triu(
                torch.ones(len(generated), len(generated), dtype=torch.bool, device=device),
                diagonal=1,
            )

            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output[:, -1, :])

            # Temperature scaling
            logits = logits / temperature

            # Greedy
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)

            if next_token == tokenizer.base_tokenizer.eos_token_id:
                break

        # Decode
        text = tokenizer.base_tokenizer.decode(
            generated, skip_special_tokens=True
        )
        return text

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_translator_step(
    translator: InternalTranslator,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    use_bf16: bool = True,
    src_padding_mask: Optional[torch.Tensor] = None,
    tgt_padding_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    单步训练翻译模型。

    Args:
        translator: InternalTranslator
        src_ids: (batch, src_len) — 内部 token 序列
        tgt_ids: (batch, tgt_len) — 目标自然语言序列
        optimizer: AdamW
        use_bf16: 是否使用 BF16

    Returns:
        loss value
    """
    translator.train()

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and src_ids.device.type == "cuda"
        else contextlib.nullcontext()
    )

    with autocast_context:
        loss = translator.compute_loss(
            src_ids,
            tgt_ids,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
    optimizer.step()

    return loss.item()
