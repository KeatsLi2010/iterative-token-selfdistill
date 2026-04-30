"""
Token 级复述生成器

核心功能：利用模型将自然语言文本"复述"为内部 token 序列。

流程：
1. 输入: <INTERNAL_START> [NL text] <INTERNAL_END>
2. 模型在 internal 区域内生成内部 token 序列
3. 输出: 完整序列 (NL prefix + internal tokens)
"""

import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
import gc


class RephraseGenerator:
    """
    批量生成内部 token 序列。

    关键参数：
    - temperature: 采样温度（高 = 多样性高，低 = 确定性高）
    - top_p: nucleus sampling
    - max_internal_tokens: 最多生成的内部 token 数
    """

    def __init__(
        self,
        model,  # SmolLM2Internal
        tokenizer,  # ExtendedTokenizer
        temperature: float = 1.2,
        top_p: float = 0.95,
        max_internal_tokens: int = 256,
        min_internal_tokens: int = 4,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.max_internal_tokens = max_internal_tokens
        self.min_internal_tokens = min_internal_tokens
        self.device = device

    def generate_single(self, text: str) -> dict:
        """
        为单个文本生成内部 token 序列。

        Args:
            text: 自然语言文本

        Returns:
            {
                "full_ids": (total_len,) — NL prefix + internal tokens
                "internal_mask": (total_len,) — True 标记 internal token 位置
                "nl_text": str — 原始文本
                "internal_token_count": int
            }
        """
        prompt_ids = self.tokenizer.build_internal_prompt(text)
        prompt_ids = prompt_ids.to(self.device)

        full_ids, internal_mask = self.model.generate_internal_tokens(
            prompt_ids=prompt_ids,
            max_internal_tokens=self.max_internal_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            min_internal_tokens=self.min_internal_tokens,
        )

        full_ids = full_ids.cpu()
        internal_mask = internal_mask.cpu()

        internal_count = internal_mask.sum().item()

        return {
            "full_ids": full_ids,
            "internal_mask": internal_mask,
            "nl_text": text,
            "internal_token_count": internal_count,
        }

    @torch.no_grad()
    def generate_batch(
        self,
        texts: list[str],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        批量为多个文本生成内部 token 序列。

        （注意：由于生成是自回归的，这里的"批量"是串行处理多个样本）

        Args:
            texts: 自然语言文本列表
            batch_size: 未使用（保留接口兼容性）
            show_progress: 是否显示进度条

        Returns:
            list of dict，每个包含 full_ids, internal_mask, nl_text
        """
        self.model.model.eval()
        results = []

        iterator = tqdm(texts, desc="Rephrasing") if show_progress else texts

        for text in iterator:
            try:
                result = self.generate_single(text)
                results.append(result)
            except Exception as e:
                print(f"[RephraseGenerator] Error on sample: {e}")
                continue

            # 定期清理 GPU 缓存
            if len(results) % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        torch.cuda.empty_cache()
        return results

    @torch.no_grad()
    def generate_from_dataset(
        self,
        samples: list[dict],
        text_key: str = "text",
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        从数据集样本批量生成内部 token 序列。

        Args:
            samples: list of dict，每个包含 text 字段
            text_key: 文本字段名
            max_samples: 最多处理多少样本
            show_progress: 是否显示进度条

        Returns:
            list of dict with internal token sequences
        """
        if max_samples is not None:
            samples = samples[:max_samples]

        texts = [s[text_key] for s in samples]
        return self.generate_batch(texts, show_progress=show_progress)

    def set_temperature(self, temperature: float):
        """动态调整采样温度。"""
        self.temperature = temperature
