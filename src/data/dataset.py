"""
数据集加载与预处理

输入: sonnet4.6 JSONL 格式
输出: 可用于训练的 tokenized sequences
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Dict
import random
import os
import pickle
from functools import partial
from multiprocessing import Pool


# ─── Worker function (module-level, needed for Windows spawn) ────────

def _tokenize_one(args):
    """单个样本的分词 worker。每个进程独立加载 tokenizer。"""
    sample, base_model, max_length = args
    try:
        messages = sample.get("messages", [])
        if not messages:
            return None

        # 独立加载 tokenizer（spawn 模式共享不了）
        from src.tokenizer.extended_tokenizer import ExtendedTokenizer
        tokenizer = ExtendedTokenizer(base_model=base_model)

        ids = tokenizer.build_chat_input(messages, max_length=max_length)
        if len(ids) < 4:
            return None

        # 转纯文本
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        text = "\n\n".join(parts)

        return {
            "input_ids": ids,
            "text": text,
            "category": sample.get("category", "general"),
        }
    except Exception:
        return None


class ChatDataset(Dataset):
    """
    从 JSONL 文件加载 chat 格式数据。

    每条数据格式:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        ...
    }

    预分词结果缓存到磁盘（.cache/），首次运行较慢，后续秒加载。
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,  # ExtendedTokenizer
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        split: str = "train",
        val_split: float = 0.05,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.val_split = val_split

        # Load raw samples
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if max_samples is not None:
            samples = samples[:max_samples]

        print(f"[ChatDataset] Loaded {len(samples)} raw samples from {jsonl_path}")

        # Split train/val
        random.seed(seed)
        random.shuffle(samples)
        val_size = int(len(samples) * val_split)

        if split == "train":
            self.samples = samples[val_size:]
        elif split == "val":
            self.samples = samples[:val_size]
        else:
            self.samples = samples

        # 缓存路径
        cache_dir = os.path.join(os.path.dirname(jsonl_path) or ".", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
        self._cache_path = os.path.join(
            cache_dir, f"{base_name}_{split}_{max_samples}.pkl")

        # 尝试从缓存加载
        if os.path.exists(self._cache_path):
            print(f"[ChatDataset] Loading tokenized data from cache: {self._cache_path}")
            with open(self._cache_path, "rb") as f:
                self._tokenized = pickle.load(f)
            print(f"[ChatDataset] {split}: {len(self._tokenized)} samples (cached)")
        else:
            # 首次运行：分块预分词 + 缓存
            print(f"[ChatDataset] Tokenizing {len(self.samples)} samples (this may take a few minutes)...")
            self._tokenized = self._tokenize_all()
            print(f"[ChatDataset] {split}: {len(self._tokenized)} samples (tokenized)")
            print(f"[ChatDataset] Saving cache to {self._cache_path}")
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _messages_to_text(self, messages: List[Dict]) -> str:
        """将 messages 转换为纯文本。"""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def _tokenize_all(self):
        """预分词所有样本（多进程并行 + 进度条）。"""
        from tqdm import tqdm

        # 准备参数：每个 sample 是一个 (sample_dict, base_model, max_length) 元组
        task_args = [
            (s, self.tokenizer.base_model, self.max_length)
            for s in self.samples
        ]

        # 多进程分词（Windows 下 spawn 模式，每个 worker 独立加载 tokenizer）
        n_workers = min(os.cpu_count() or 4, 8)
        tokenized = []
        with Pool(processes=n_workers) as pool:
            with tqdm(total=len(task_args), desc="Tokenizing",
                      unit="samples", mininterval=1.0) as pbar:
                for result in pool.imap_unordered(_tokenize_one, task_args, chunksize=200):
                    if result is not None:
                        tokenized.append(result)
                    pbar.update(1)

        return tokenized

    def __len__(self):
        return len(self._tokenized)

    def __getitem__(self, idx):
        return self._tokenized[idx]


class InternalSequenceDataset(Dataset):
    """
    存储内部 token 序列，用于迭代训练。

    格式: <INTERNAL_START> [NL text] <INTERNAL_END> [internal tokens] <INTERNAL_END>

    训练时只计算 internal token 部分的 loss。
    """

    def __init__(
        self,
        sequences: List[Dict],  # [{"full_ids": ..., "internal_mask": ..., "nl_text": ...}]
        max_length: int = 2048,
    ):
        self.max_length = max_length
        self.data = []

        for seq in sequences:
            full_ids = seq["full_ids"]
            internal_mask = seq.get("internal_mask", None)

            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                if internal_mask is not None:
                    internal_mask = internal_mask[:max_length]

            # Auto-generate internal mask if not provided
            if internal_mask is None:
                internal_mask = torch.zeros(len(full_ids), dtype=torch.bool)

            self.data.append({
                "input_ids": full_ids,
                "internal_mask": internal_mask,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"]
        internal_mask = item["internal_mask"]
        seq_len = len(input_ids)

        # Labels: same as input_ids but mask non-internal positions
        labels = input_ids.clone()
        labels[~internal_mask] = -100  # ignore in loss

        return {
            "input_ids": input_ids,
            "labels": labels,
            "internal_mask": internal_mask,
        }


def collate_chat_batch(batch: List[Dict], pad_token_id: int) -> Dict:
    """批处理 collator for ChatDataset。"""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        seq_len = ids.size(0)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
        labels[i, :seq_len] = ids  # standard LM: predict next token
        labels[i, 0] = -100  # don't predict first token

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_internal_batch(batch: List[Dict], pad_token_id: int) -> Dict:
    """批处理 collator for InternalSequenceDataset。"""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        seq_len = ids.size(0)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
        labels[i, :seq_len] = ids
        labels[i, 0] = -100

        # Only compute loss on internal token positions
        imask = item["internal_mask"]
        loss_mask[i, :seq_len] = imask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }


def create_dataloaders(
    jsonl_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
    val_split: float = 0.05,
    num_workers: int = 2,
) -> tuple:
    """创建 train/val dataloaders。"""
    train_ds = ChatDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        split="train",
        val_split=val_split,
    )
    val_ds = ChatDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        split="val",
        val_split=val_split,
    )

    pad_id = tokenizer.pad_id

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_chat_batch, pad_token_id=pad_id),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_chat_batch, pad_token_id=pad_id),
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
