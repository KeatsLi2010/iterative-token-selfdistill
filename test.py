#!/usr/bin/env python3
"""
模型测试与评估工具

用法:
    # 基本推理测试
    python test.py --checkpoint checkpoints/round_0

    # 生成文本
    python test.py --checkpoint checkpoints/round_0 --prompt "What is machine learning?"

    # 复述测试（自然语言 → 内部 token）
    python test.py --checkpoint checkpoints/round_0 --rephrase "The cat sat on the mat."

    # 评估困惑度
    python test.py --checkpoint checkpoints/round_0 --eval data/test.jsonl

    # 交互模式
    python test.py --checkpoint checkpoints/round_0 --interactive

    # 对比多个 checkpoint
    python test.py --compare checkpoints/round_0 checkpoints/round_1

    # 测试翻译模型
    python test.py --checkpoint checkpoints/round_5 --test-translator
"""

import argparse
import sys
import os
import json
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tokenizer.extended_tokenizer import ExtendedTokenizer
from src.model.smollm2_internal import SmolLM2Internal
from src.trainer.rephrase import RephraseGenerator
from src.eval.metrics import compute_metrics, print_metrics


# ─── Helper ───────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载 checkpoint 模型 + tokenizer（含训练后的完整词表状态）。"""
    import pickle  # may be needed for tokenizer deserialization

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Tokenizer: 优先从 checkpoint 加载（含训练时添加的内部 token）
    if (ckpt / "tokenizer.json").exists() and (ckpt / "tokenizer_ext.pt").exists():
        print(f"[Test] Loading tokenizer from checkpoint: {checkpoint_path}")
        tokenizer = ExtendedTokenizer(base_model=str(ckpt))
        # 验证词表一致性
        ext_meta = torch.load(str(ckpt / "tokenizer_ext.pt"), map_location="cpu")
        assert tokenizer.vocab_size == ext_meta["vocab_size"], \
            f"Vocab mismatch: {tokenizer.vocab_size} vs {ext_meta['vocab_size']}"
        print(f"[Test] Tokenizer vocab: {tokenizer.vocab_size}")
    else:
        # Fallback: 从原始 SmolLM2 创建（仅用于无 checkpoint 的测试）
        local_smollm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SmolLM2-135M-Instruct")
        if os.path.isdir(local_smollm):
            tokenizer_model = local_smollm
        else:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            tokenizer_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
        print(f"[Test] Loading tokenizer from base model: {tokenizer_model}")
        tokenizer = ExtendedTokenizer(base_model=tokenizer_model)

    # 加载模型权重
    has_ckpt = (ckpt / "extended_config.pt").exists()
    print(f"[Test] Loading model from: {checkpoint_path}" + (" (full ckpt)" if has_ckpt else ""))
    if has_ckpt:
        model = SmolLM2Internal.from_checkpoint(
            str(ckpt), device=device, use_gradient_checkpointing=False,
        )
    else:
        # Fallback: 创建未训练的模型
        local_smollm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SmolLM2-135M-Instruct")
        tokenizer_model = local_smollm if os.path.isdir(local_smollm) else "HuggingFaceTB/SmolLM2-135M-Instruct"
        model = SmolLM2Internal(
            base_model=tokenizer_model,
            n_internal_tokens=4096,
            n_special_tokens=8,
            use_gradient_checkpointing=False,
            device=device,
        )

    model.model.eval()
    return model, tokenizer


def load_translator(checkpoint_path: str, device: str = "cuda"):
    """加载翻译模型。"""
    from src.translator.model import InternalTranslator

    pt_path = os.path.join(checkpoint_path, "translator.pt")
    if not os.path.exists(pt_path):
        print(f"[Test] Translator not found at {pt_path}")
        return None

    translator = InternalTranslator(
        src_vocab_size=53256,
        tgt_vocab_size=49152,
        d_model=384,
        n_heads=6,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=1024,
    )
    translator.load_state_dict(torch.load(pt_path, map_location=device))
    translator.to(device)
    translator.eval()
    print(f"[Test] Translator loaded from {pt_path}")
    return translator


# ─── Inference ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """标准自然语言文本生成。"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, max_length=1024)
    input_ids = input_ids.unsqueeze(0).to(device)

    generated = model.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.base_tokenizer.decode(generated[0], skip_special_tokens=True)
    return full_text


@torch.no_grad()
def rephrase_text(
    model, tokenizer, text: str,
    temperature: float = 1.2,
    top_p: float = 0.95,
    max_internal: int = 256,
    device: str = "cuda",
) -> dict:
    """将自然语言文本复述为内部 token 序列。"""
    gen = RephraseGenerator(
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=top_p,
        max_internal_tokens=max_internal,
        device=device,
    )
    return gen.generate_single(text)


def format_rephrase_result(result: dict, tokenizer) -> str:
    """格式化复述结果，显示内部 token。"""
    full_ids = result["full_ids"]
    mask = result["internal_mask"]
    nl_part = full_ids[~mask]
    internal_part = full_ids[mask]

    lines = []
    lines.append(f"原文: {result['nl_text']}")
    lines.append(f"NL tokens ({len(nl_part)}): {nl_part.tolist()}")
    lines.append(f"Internal tokens ({len(internal_part)}): {internal_part.tolist()}")
    lines.append(f"Internal token count: {result['internal_token_count']}")
    return "\n".join(lines)


# ─── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model, tokenizer, texts: List[str],
    max_length: int = 512,
    device: str = "cuda",
) -> float:
    """计算文本列表的困惑度。"""
    model.model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_length)
        if len(ids) < 2:
            continue

        ids = ids.unsqueeze(0).to(device)
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100

        outputs = model.model(input_ids=ids, labels=labels)
        loss = outputs.loss
        n_tokens = (labels != -100).sum().item()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def evaluate_checkpoint(
    checkpoint_path: str,
    texts: Optional[List[str]] = None,
    device: str = "cuda",
) -> Dict:
    """综合评估一个 checkpoint。"""
    model, tokenizer = load_model(checkpoint_path, device)
    results = {"checkpoint": checkpoint_path}

    # 模型参数
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    results["params_M"] = n_params
    print(f"  Params: {n_params:.1f}M")

    # 困惑度
    if texts:
        ppl = compute_perplexity(model, tokenizer, texts, device=device)
        results["perplexity"] = ppl
        print(f"  Perplexity: {ppl:.2f}")

    # 内部 token 测试
    if texts:
        print(f"  Testing rephrase on {min(10, len(texts))} samples...")
        gen = RephraseGenerator(
            model=model, tokenizer=tokenizer,
            temperature=1.0, device=device,
        )
        internal_seqs = []
        nl_counts = []
        for text in texts[:10]:
            try:
                result = gen.generate_single(text)
                mask = result["internal_mask"]
                internal_part = result["full_ids"][mask]
                nl_part = result["full_ids"][~mask]
                internal_seqs.append(internal_part)
                nl_counts.append(len(nl_part))
            except Exception as e:
                print(f"    Error: {e}")

        if internal_seqs:
            metrics = compute_metrics(
                internal_sequences=internal_seqs,
                nl_token_counts=nl_counts,
                n_internal_tokens=4096,
                internal_base_id=model.internal_base_id,
            )
            results.update(metrics)

    return results


# ─── Translator Test ──────────────────────────────────────────────────

@torch.no_grad()
def test_translator(
    checkpoint_path: str,
    texts: List[str],
    device: str = "cuda",
):
    """端到端测试：NL → 内部 token → NL 翻译。"""
    model, tokenizer = load_model(checkpoint_path, device)
    translator = load_translator(checkpoint_path, device)

    if translator is None:
        print("[Test] No translator found, skipping translation test.")
        return

    print(f"\n{'='*60}")
    print(f"  Translator Test: NL → Internal → NL")
    print(f"{'='*60}")

    for i, text in enumerate(texts[:5]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input:      {text[:100]}...")

        # Step 1: NL → Internal tokens
        result = rephrase_text(model, tokenizer, text, device=device)
        mask = result["internal_mask"]
        internal_ids = result["full_ids"][mask]
        internal_ids = internal_ids.unsqueeze(0).to(device)

        # Step 2: Internal → NL (translator)
        tgt_prefix = torch.tensor([[tokenizer.nl_start_id]], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            translated_ids = translator.generate(
                src=internal_ids,
                tgt_prefix=tgt_prefix,
                max_len=256,
                eos_id=tokenizer.nl_end_id,
            )
        translated_text = tokenizer.decode(
            translated_ids[0].cpu(), skip_special_tokens=True,
        )
        print(f"Translated:  {translated_text[:200]}")


# ─── Compare Checkpoints ──────────────────────────────────────────────

def compare_checkpoints(
    checkpoint_paths: List[str],
    texts: Optional[List[str]] = None,
    device: str = "cuda",
):
    """对比多个 checkpoint 的指标。"""
    print(f"\n{'='*60}")
    print(f"  Comparing {len(checkpoint_paths)} Checkpoints")
    print(f"{'='*60}")

    all_results = []
    for cp in checkpoint_paths:
        print(f"\n[{cp}]")
        result = evaluate_checkpoint(cp, texts=texts, device=device)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    header = f"{'Checkpoint':<30} {'Params':>8} {'PPL':>8} {'Entropy':>8} {'Usage':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        cp = os.path.basename(r["checkpoint"])[:28]
        params = f"{r.get('params_M', 0):.1f}M"
        ppl = f"{r.get('perplexity', float('nan')):.1f}"
        ent = f"{r.get('entropy', float('nan')):.2f}"
        usage = r.get("effective_token_usage", {})
        ur = f"{usage.get('usage_rate', 0):.1%}" if usage else "N/A"
        print(f"{cp:<30} {params:>8} {ppl:>8} {ent:>8} {ur:>8}")


# ─── Interactive ──────────────────────────────────────────────────────

def interactive_mode(checkpoint_path: str, device: str = "cuda"):
    """交互式测试模式。"""
    model, tokenizer = load_model(checkpoint_path, device)
    translator = load_translator(checkpoint_path, device)

    print(f"\n{'='*60}")
    print(f"  Interactive Test Mode")
    print(f"  Model: {checkpoint_path}")
    print(f"  Commands:")
    print(f"    /gen <prompt>     — 生成文本")
    print(f"    /rephrase <text>  — 复述为内部 token")
    print(f"    /translate <text> — NL→内部→NL 翻译")
    print(f"    /temp <value>     — 设置采样温度")
    print(f"    /quit             — 退出")
    print(f"{'='*60}\n")

    temp = 0.8

    while True:
        try:
            cmd = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not cmd:
            continue

        if cmd.startswith("/quit"):
            break

        elif cmd.startswith("/temp"):
            try:
                temp = float(cmd.split()[1])
                print(f"Temperature set to {temp}")
            except (IndexError, ValueError):
                print(f"Current temperature: {temp}")

        elif cmd.startswith("/gen"):
            prompt = cmd[len("/gen "):]
            response = generate_text(
                model, tokenizer, prompt,
                temperature=temp, device=device,
            )
            print(f"\n{response}\n")

        elif cmd.startswith("/rephrase"):
            text = cmd[len("/rephrase "):]
            result = rephrase_text(
                model, tokenizer, text,
                temperature=temp, device=device,
            )
            print(format_rephrase_result(result, tokenizer))

        elif cmd.startswith("/translate"):
            text = cmd[len("/translate "):]
            if translator is None:
                print("No translator loaded.")
                continue
            result = rephrase_text(
                model, tokenizer, text,
                temperature=temp, device=device,
            )
            mask = result["internal_mask"]
            internal_ids = result["full_ids"][mask].unsqueeze(0).to(device)

            tgt_prefix = torch.tensor([[tokenizer.nl_start_id]], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                translated_ids = translator.generate(
                    src=internal_ids,
                    tgt_prefix=tgt_prefix,
                    max_len=256,
                    eos_id=tokenizer.nl_end_id,
                )
            translated = tokenizer.decode(translated_ids[0].cpu(), skip_special_tokens=True)
            print(f"\nInput: {text[:200]}")
            print(f"Internal: {internal_ids[0].cpu().tolist()}")
            print(f"Translated: {translated}\n")

        else:
            # Default: generate from prompt
            response = generate_text(
                model, tokenizer, cmd,
                temperature=temp, device=device,
            )
            print(f"\n{response}\n")


# ─── Main ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Model Testing & Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint directory path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for text generation")
    parser.add_argument("--rephrase", type=str, default=None,
                        help="Text to rephrase into internal tokens")
    parser.add_argument("--eval", type=str, default=None,
                        help="JSONL file for perplexity evaluation")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                        help="Compare multiple checkpoints")
    parser.add_argument("--test-translator", action="store_true",
                        help="Test translator with sample texts")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive test mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--rephrase-temp", type=float, default=1.2,
                        help="Temperature for rephrase sampling")
    return parser.parse_args()


def main():
    import math  # needed for perplexity

    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Compare mode ──
    if args.compare:
        texts = None
        if args.eval:
            with open(args.eval, "r", encoding="utf-8") as f:
                texts = [json.loads(line).get("text", "") for line in f if line.strip()][:100]
        compare_checkpoints(args.compare, texts=texts, device=args.device)
        return

    # ── Need checkpoint ──
    if not args.checkpoint:
        print("Error: --checkpoint required (unless using --compare)")
        sys.exit(1)

    # ── Interactive mode ──
    if args.interactive:
        interactive_mode(args.checkpoint, device=args.device)
        return

    # ── Load model ──
    model, tokenizer = load_model(args.checkpoint, device=args.device)

    # ── Single prompt generation ──
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        print(text)
        return

    # ── Single rephrase ──
    if args.rephrase:
        result = rephrase_text(
            model, tokenizer, args.rephrase,
            temperature=args.rephrase_temp,
            device=args.device,
        )
        print(format_rephrase_result(result, tokenizer))
        return

    # ── Evaluation ──
    if args.eval:
        print(f"\nLoading evaluation data from: {args.eval}")
        with open(args.eval, "r", encoding="utf-8") as f:
            texts = [json.loads(line).get("text", "") for line in f if line.strip()]
        print(f"  Loaded {len(texts)} texts")

        result = evaluate_checkpoint(
            args.checkpoint, texts=texts[:500], device=args.device,
        )
        print(f"\nResults:")
        for k, v in result.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {k}: {v}")
        return

    # ── Translator test ──
    if args.test_translator:
        # Load some sample texts
        import random
        sample_texts = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The quick brown fox jumps over the lazy dog.",
            "Python is a high-level programming language known for its readability.",
            "Deep neural networks use multiple layers to progressively extract higher-level features.",
            "Reinforcement learning agents learn by interacting with an environment and receiving rewards.",
        ]
        test_translator(args.checkpoint, sample_texts, device=args.device)
        return

    # ── Default: quick evaluation ──
    print("\nNo action specified. Running quick evaluation...")
    result = evaluate_checkpoint(args.checkpoint, texts=None, device=args.device)
    print(f"\nModel: {os.path.basename(args.checkpoint)}")
    print(f"Params: {result.get('params_M', 'N/A'):.1f}M")
    print("\nTip: Use --prompt, --rephrase, --eval, --test-translator, or --interactive")


if __name__ == "__main__":
    main()
