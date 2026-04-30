"""
评估指标

- Token 熵：衡量内部 token 分布的多样性
- 有效 token 使用率：4096 个 internal token 中实际使用了多少
- 翻译 BLEU：内部语言→自然语言的翻译质量
"""

import torch
import torch.nn.functional as F
from collections import Counter
import math
from typing import List, Dict, Optional


def compute_token_entropy(
    internal_sequences: List[torch.Tensor],
    n_internal_tokens: int = 4096,
    internal_base_id: int = 49154,
) -> float:
    """
    计算内部 token 分布的熵。

    Args:
        internal_sequences: list of internal token ID tensors
        n_internal_tokens: 内部 token 总数
        internal_base_id: 第一个内部 token 的 ID

    Returns:
        entropy (bits)
    """
    if not internal_sequences:
        return 0.0

    # Count token frequencies
    counter = Counter()
    for seq in internal_sequences:
        for tid in seq.tolist():
            idx = tid - internal_base_id
            if 0 <= idx < n_internal_tokens:
                counter[idx] += 1

    total = sum(counter.values())
    if total == 0:
        return 0.0

    # Compute entropy
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def compute_effective_token_usage(
    internal_sequences: List[torch.Tensor],
    n_internal_tokens: int = 4096,
    internal_base_id: int = 49154,
) -> Dict:
    """
    计算内部 token 使用率。

    Returns:
        {
            "used_count": 实际使用的 token 种类数,
            "total_available": 4096,
            "usage_rate": 使用率 (0-1),
            "top10_ids": 使用频率最高的 10 个 token ID,
            "top10_freqs": 对应频率,
        }
    """
    counter = Counter()
    for seq in internal_sequences:
        for tid in seq.tolist():
            idx = tid - internal_base_id
            if 0 <= idx < n_internal_tokens:
                counter[idx] += 1

    used = len(counter)
    usage_rate = used / n_internal_tokens

    top10 = counter.most_common(10)
    top10_ids = [t[0] for t in top10]
    top10_freqs = [t[1] for t in top10]

    return {
        "used_count": used,
        "total_available": n_internal_tokens,
        "usage_rate": usage_rate,
        "top10_ids": top10_ids,
        "top10_freqs": top10_freqs,
    }


def compute_compression_ratio(
    internal_sequences: List[torch.Tensor],
    nl_token_counts: List[int],
) -> float:
    """
    计算压缩率 = 内部 token 数 / 原文 token 数。

    < 1.0 表示内部语言比原文更紧凑。
    """
    total_internal = sum(len(seq) for seq in internal_sequences)
    total_nl = sum(nl_token_counts)
    if total_nl == 0:
        return 1.0
    return total_internal / total_nl


def compute_metrics(
    internal_sequences: List[torch.Tensor],
    nl_token_counts: Optional[List[int]] = None,
    n_internal_tokens: int = 4096,
    internal_base_id: int = 49154,
) -> Dict:
    """
    综合评估指标。

    Returns:
        {
            "entropy": float,
            "effective_token_usage": Dict,
            "compression_ratio": float (如果提供了 nl_token_counts),
            "total_samples": int,
            "avg_internal_tokens_per_sample": float,
        }
    """
    metrics = {
        "entropy": compute_token_entropy(
            internal_sequences, n_internal_tokens, internal_base_id
        ),
        "effective_token_usage": compute_effective_token_usage(
            internal_sequences, n_internal_tokens, internal_base_id
        ),
        "total_samples": len(internal_sequences),
        "avg_internal_tokens_per_sample": (
            sum(len(s) for s in internal_sequences) / max(1, len(internal_sequences))
        ),
    }

    if nl_token_counts is not None:
        metrics["compression_ratio"] = compute_compression_ratio(
            internal_sequences, nl_token_counts
        )

    return metrics


def print_metrics(metrics: Dict, round_num: int):
    """格式化打印指标。"""
    print(f"\n{'='*60}")
    print(f"  Round {round_num} Metrics")
    print(f"{'='*60}")
    print(f"  Samples:          {metrics['total_samples']}")
    print(f"  Avg internal len: {metrics['avg_internal_tokens_per_sample']:.1f}")
    print(f"  Token entropy:    {metrics['entropy']:.3f} bits")
    usage = metrics['effective_token_usage']
    print(f"  Token usage:      {usage['used_count']}/{usage['total_available']} "
          f"({usage['usage_rate']:.1%})")
    if 'compression_ratio' in metrics:
        print(f"  Compression:      {metrics['compression_ratio']:.3f}")
    print(f"{'='*60}\n")
