#!/usr/bin/env python3
"""
迭代式 Token 级自蒸馏语言演化实验

用法:
    # 完整运行（20 轮）
    python train.py

    # 仅运行 Round 0（基线）
    python train.py --rounds 0

    # 自定义配置
    python train.py --config configs/smollm2.yaml --rounds 10 --batch-size 4

    # 从 checkpoint 继续
    python train.py --resume checkpoints/round_5
"""

import argparse
import sys
import os
import yaml
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path: str) -> dict:
    """加载 YAML 配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative Token-Level Self-Distillation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/smollm2.yaml",
        help="YAML config path"
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Override n_rounds (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max steps per round"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to load from JSONL"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda / cpu)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint directory"
    )
    parser.add_argument(
        "--round-0-only", action="store_true",
        help="Only run round 0 (baseline)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Local model path or HF model name (default: SmolLM2-135M-Instruct local or mirror)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override from CLI
    if args.rounds is not None:
        config.setdefault("iterative", {})["n_rounds"] = args.rounds
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps_per_round"] = args.max_steps
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples

    if args.round_0_only:
        config.setdefault("iterative", {})["n_rounds"] = 0

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Try local SmolLM2 directory first
        local_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SmolLM2-135M-Instruct")
        if os.path.isdir(local_model):
            model_path = local_model
            print(f"[Info] Using local model: {local_model}")
        else:
            # Use HF mirror for mainland China
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"
            print(f"[Info] Model not found locally, using HF mirror")

    # Dry run
    if args.dry_run:
        print("=== Config ===")
        print(yaml.dump(config, default_flow_style=False))
        print("\n=== Device ===")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"\n=== Model Path ===")
        print(f"  {model_path}")
        return

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  PyTorch: {torch.__version__}")

    # Initialize tokenizer
    from src.tokenizer.extended_tokenizer import ExtendedTokenizer
    from src.trainer.iterative_trainer import IterativeTrainer

    print("\n[Init] Loading tokenizer...")
    tokenizer = ExtendedTokenizer(
        base_model=model_path
    )

    # Create trainer
    trainer = IterativeTrainer(
        config=config,
        tokenizer=tokenizer,
        device=args.device,
        model_path=model_path,
    )

    # Run
    trainer.run()


if __name__ == "__main__":
    main()
