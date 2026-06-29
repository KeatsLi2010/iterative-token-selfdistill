# Iterative Token-Level Self-Distillation

本项目探索一个问题：大语言模型能否在自然语言之外，学习一种更压缩、更稳定、对任务更有用、且仍然可解释的离散内部表达。

项目当前的核心不是让模型生成一串难以理解的“乱码 token”，而是把 internal token 视为模型在回答前生成的离散内部推理草稿。这个草稿需要承担三个功能：压缩输入信息、保留任务相关结构、通过翻译器保留可解释出口。

## 研究假设

自然语言适合人类交流，但不一定是模型进行中间推理的最高效表示。如果给模型一组额外的离散 internal tokens，并通过多轮生成、训练和评估施加约束，模型可能逐步形成一种介于自然语言和隐藏向量之间的中间表示协议。

本项目关注的不是“internal token 看起来像不像语言”，而是它是否具备以下性质：

- 压缩性：internal token 序列是否能比原文更短或更集中地表达关键信息。
- 稳定性：相同或相近输入是否能产生相对一致的 internal 表达。
- 功能性：internal token 是否能辅助后续训练、生成和推理。
- 可解释性：internal token 是否能通过 translator 还原为自然语言摘要、推理线索或关键信息。

## 设计原则

### 1. Internal token 是推理草稿

早期版本将流程称为 rephrase，容易被理解为“把自然语言换一种 token 复述出来”。现在更准确的定位是：

```text
自然语言输入 x -> 离散内部草稿 z -> 训练/生成/解释
```

`z` 不应该完整复制原文，而应当逐步学会保留对任务有用的信息。

### 2. 信息瓶颈优先于无限表达

如果 internal token 长度不受限制，模型可能只是把原文编码成另一套符号。项目因此加入长度感知的生成上限：每条样本的 internal token 数量由原文长度和全局上限共同决定。

默认配置中：

```yaml
max_internal_tokens_multiplier: 1.0
max_internal_tokens_cap: 256
```

这意味着 internal 草稿不会无条件扩张，而是被鼓励形成更紧凑的表达。

### 3. 多样性需要受控

Entropy bonus 的目标不是让 token 越随机越好，而是避免模型塌缩到少数 token，同时观察 token 使用是否形成稳定结构。更理想的方向是 controlled diversity：既防止塌缩，也避免纯噪声。

### 4. Translator 是可解释性约束

translator 不只是附属翻译模型，而是防止 internal token 退化为不可解释噪声的约束器。它尝试把 internal 表达还原为自然语言，从而让研究者能够检查模型到底保留了哪些信息。

## 系统架构

```text
JSONL chat 数据
  -> ExtendedTokenizer
  -> SmolLM2Internal
  -> Round 0 自然语言 baseline fine-tune
  -> Round K internal draft generation
  -> InternalSequenceDataset + 原始数据混合
  -> SingleRoundTrainer 继续训练主模型
  -> InternalTranslator 学习 internal -> natural language
  -> Metrics 统计 entropy / token usage / compression
  -> Checkpoint 保存
```

## 主要模块

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 训练入口 | `train.py` | 读取配置、初始化 tokenizer 和 trainer |
| 配置 | `configs/smollm2.yaml` | 定义模型、训练、internal token、日志路径 |
| 扩展 tokenizer | `src/tokenizer/extended_tokenizer.py` | 在 SmolLM2 原词表上追加 internal/special tokens |
| 主模型 | `src/model/smollm2_internal.py` | 包装 SmolLM2，扩展 embedding，并限制 internal token 采样范围 |
| 数据集 | `src/data/dataset.py` | 加载 chat JSONL、缓存 tokenized 数据、构造 internal 序列数据 |
| 迭代训练器 | `src/trainer/iterative_trainer.py` | 编排 Round 0 和 Round K 的生成、混合、训练、评估、保存 |
| 单轮训练 | `src/trainer/training_loop.py` | AdamW、warmup/cosine、梯度累积、bf16、entropy bonus |
| internal 生成 | `src/trainer/rephrase.py` | 生成离散 internal draft，并按输入长度控制上限 |
| 翻译器 | `src/translator/model.py` | Encoder-Decoder Transformer，将 internal token 翻译回自然语言 |
| 指标 | `src/eval/metrics.py` | 统计 token entropy、有效 token 使用率、压缩率 |
| 测试工具 | `test.py` | checkpoint 加载、文本生成、internal 生成、translator 测试 |

## 训练流程

```text
Round 0:
  使用自然语言 chat 数据对 SmolLM2Internal 做 baseline fine-tune。

Round K:
  1. 使用当前模型为部分样本生成 internal draft。
  2. 按长度瓶颈限制 internal token 数量。
  3. 将 internal 数据与原始自然语言数据混合。
  4. 继续训练主模型，并对 internal 区域计算专门 loss。
  5. 训练 translator，保留 internal -> natural language 的解释通道。
  6. 评估 entropy、token usage、compression ratio。
  7. 保存 round/step 级 checkpoint。
```

## 快速开始

```bash
pip install -r requirements.txt

# 查看配置和设备，不加载大模型
python train.py --dry-run --device cpu --max-samples 1 --max-steps 1 --round-0-only

# 只跑 Round 0 baseline
python train.py --round-0-only --max-steps 500 --max-samples 5000

# 完整迭代训练
python train.py

# 从 checkpoint 测试
python test.py --checkpoint checkpoints/round_0 --prompt "What is machine learning?"
```

## 当前状态

已完成：

- SmolLM2-135M-Instruct 加载与词表扩展。
- 4096 个 internal tokens 与特殊控制 tokens。
- ChatDataset / InternalSequenceDataset。
- 多轮迭代训练框架。
- internal token 生成器。
- translator 模型。
- entropy、token usage、compression 等基础指标。
- Windows 兼容、checkpoint 保存、CPU dry-run、bf16 设备兼容修复。
- internal token 生成的长度感知上限。

仍在推进：

- Round 0 baseline 训练稳定性验证。
- internal draft 的质量筛选。
- 更明确的 controlled diversity 目标。
- translator 输出从“完整复原文本”转向“摘要/推理线索/关键信息”。

## 项目定位

本项目是一个研究性原型，重点在于验证“离散内部推理草稿”这一方向是否有实验价值。短期目标不是追求下游榜单成绩，而是建立一个可运行、可观测、可解释的实验框架，为后续更严格的对照实验和任务评估打基础。
