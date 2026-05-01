"""
扩展分词器：在 SmolLM2 tokenizer 基础上添加内部离散 token 词汇表

核心设计：
- 保留原始 49152 tokens 不变（保证 SmolLM2 权重可直接加载）
- 追加 4096 个 <INTERNAL_i> tokens（模型在这些 token 上自由采样）
- 追加 8 个特殊控制 tokens
- 总计 53256 tokens
"""

import torch
from transformers import AutoTokenizer


class ExtendedTokenizer:
    """
    扩展分词器，在原始 tokenizer 基础上添加内部 token 词汇表。

    特殊 token 布局（id 范围）：
      <INTERNAL_START>  = 49152   # 标记内部语言开始
      <INTERNAL_END>    = 49153   # 标记内部语言结束
      <INTERNAL_0>      = 49154   # 第一个内部 token
      ...
      <INTERNAL_4095>   = 53249   # 最后一个内部 token
      <NL_START>        = 53250   # 自然语言开始
      <NL_END>          = 53251   # 自然语言结束
      <PAD>             = 53252   # 填充
      <TASK_MATH>       = 53253   # 数学推理
      <TASK_LOGIC>      = 53254   # 逻辑推理
      <TASK_CODE>       = 53255   # 代码生成
    """

    # 特殊 token 名称与 ID 的映射
    SPECIAL_TOKENS = {
        "internal_start": ("<INTERNAL_START>", 49152),
        "internal_end":   ("<INTERNAL_END>",   49153),
        "nl_start":       ("<NL_START>",       53250),
        "nl_end":         ("<NL_END>",         53251),
        "pad":            ("<PAD>",            53252),
        "task_math":      ("<TASK_MATH>",      53253),
        "task_logic":     ("<TASK_LOGIC>",     53254),
        "task_code":      ("<TASK_CODE>",      53255),
    }

    INTERNAL_BASE_ID = 49154
    N_INTERNAL_TOKENS = 4096
    INTERNAL_END_ID = 53249      # INTERNAL_BASE_ID + 4096 - 1
    TOTAL_VOCAB_SIZE = 53256     # 49152 + 4096 + 8
    ORIGINAL_VOCAB_SIZE = 49152

    def __init__(self, base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
                 verbose: bool = True):
        """
        Args:
            base_model: HuggingFace 模型名，用于加载原始 tokenizer
            verbose: 是否打印初始化日志（多进程 worker 中应关闭）
        """
        self.base_model = base_model
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._ensure_pad_token()

        # Token 添加顺序（必须与 config/smollm2.yaml 中的 ID 布局一致）：
        #   <INTERNAL_START>  = 49152
        #   <INTERNAL_END>    = 49153
        #   <INTERNAL_0>      = 49154
        #   ...
        #   <INTERNAL_4095>   = 53249
        #   <NL_START>        = 53250
        #   <NL_END>          = 53251
        #   <PAD>             = 53252
        #   <TASK_MATH>       = 53253
        #   <TASK_LOGIC>      = 53254
        #   <TASK_CODE>       = 53255
        internal_tokens = [
            f"<INTERNAL_{i}>" for i in range(self.N_INTERNAL_TOKENS)
        ]
        # 先添加 INTERNAL_START + INTERNAL_END，再添加内部 token，最后添加其余特殊 token
        ordered_tokens = [
            "<INTERNAL_START>",
            "<INTERNAL_END>",
        ] + internal_tokens + [
            "<NL_START>",
            "<NL_END>",
            "<PAD>",
            "<TASK_MATH>",
            "<TASK_LOGIC>",
            "<TASK_CODE>",
        ]

        num_added = self.base_tokenizer.add_tokens(ordered_tokens)
        if verbose:
            print(f"[ExtendedTokenizer] Added {num_added} tokens "
                  f"(internal={len(internal_tokens)}, special=8)  "
                  f"→ total vocab: {len(self.base_tokenizer)}")

        # Internal token ID 范围
        self.internal_start_id = self.token_to_id("<INTERNAL_START>")
        self.internal_end_id = self.token_to_id("<INTERNAL_END>")
        self.internal_ids = [
            self.token_to_id(f"<INTERNAL_{i}>")
            for i in range(self.N_INTERNAL_TOKENS)
        ]
        self.nl_start_id = self.token_to_id("<NL_START>")
        self.nl_end_id = self.token_to_id("<NL_END>")
        self.pad_id = self.token_to_id("<PAD>")
        self.task_ids = {
            "math":  self.token_to_id("<TASK_MATH>"),
            "logic": self.token_to_id("<TASK_LOGIC>"),
            "code":  self.token_to_id("<TASK_CODE>"),
        }

        # 构建 internal token mask（用于限制采样范围）
        self.internal_token_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for iid in self.internal_ids:
            self.internal_token_mask[iid] = True
        self.internal_token_mask[self.internal_end_id] = True

        # 自然语言 token mask（原始 token + NL_END）
        self.nl_token_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.nl_token_mask[:self.ORIGINAL_VOCAB_SIZE] = True
        self.nl_token_mask[self.nl_end_id] = True
        self.nl_token_mask[self.pad_id] = True

    def _ensure_pad_token(self):
        """确保 tokenizer 有 pad_token"""
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return len(self.base_tokenizer)

    @property
    def eos_token_id(self) -> int:
        return self.base_tokenizer.eos_token_id

    def token_to_id(self, token: str) -> int:
        """token 字符串 → token ID"""
        tid = self.base_tokenizer.convert_tokens_to_ids(token)
        if tid == self.base_tokenizer.unk_token_id:
            raise ValueError(f"Token '{token}' not found in vocab")
        return tid

    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: int = 2048, truncation: bool = True) -> torch.Tensor:
        """编码自然语言文本 → token IDs"""
        tokens = self.base_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0)

    def decode(self, token_ids: torch.Tensor,
               skip_special_tokens: bool = True) -> str:
        """token IDs → 自然语言文本"""
        return self.base_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def build_chat_input(
        self,
        messages: list[dict],
        max_length: int = 2048,
    ) -> torch.Tensor:
        """
        将 chat 格式的消息转换为 token 序列。

        格式: <NL_START> system_msg user_msg assistant_msg <NL_END>

        Args:
            messages: [{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."},
                       {"role": "assistant", "content": "..."}]
        Returns:
            token_ids: (seq_len,) tensor
        """
        parts = [self.nl_start_id]
        for msg in messages:
            parts.extend(self.encode(
                msg["content"], add_special_tokens=False,
                max_length=max_length // len(messages), truncation=True
            ).tolist())
        parts.append(self.nl_end_id)

        ids = torch.tensor(parts, dtype=torch.long)
        if len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def build_internal_prompt(self, text: str) -> torch.Tensor:
        """
        构建内部语言 prompt：
        <INTERNAL_START> [原文] <INTERNAL_END>
        模型需要在 START/END 之间采样内部 token
        """
        text_ids = self.encode(text, add_special_tokens=False,
                               max_length=2048 - 2, truncation=True)
        prompt = torch.cat([
            torch.tensor([self.internal_start_id]),
            text_ids,
            torch.tensor([self.internal_end_id]),
        ])
        return prompt

    def restrict_to_internal(self, logits: torch.Tensor) -> torch.Tensor:
        """
        限制 logits 只允许采样内部 token（mask 掉自然语言 token）。

        Args:
            logits: (batch, vocab_size) or (vocab_size,)
        Returns:
            masked logits
        """
        mask = self.internal_token_mask.to(logits.device)
        return logits.masked_fill(~mask, float("-inf"))

    def restrict_to_nl(self, logits: torch.Tensor) -> torch.Tensor:
        """限制 logits 只允许采样自然语言 token"""
        mask = self.nl_token_mask.to(logits.device)
        return logits.masked_fill(~mask, float("-inf"))

    def get_trainable_embedding_indices(self) -> torch.Tensor:
        """
        返回需要训练的 embedding 索引。
        只训练 internal tokens 对应的 embedding，原始 token embedding 冻结。
        """
        return torch.tensor(self.internal_ids + [self.internal_start_id,
                                                  self.internal_end_id])
