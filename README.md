# 迭代式 Token 级自蒸馏语言演化实验

## 一句话摘要

训练 **SmolLM2-135M-Instruct**（~140M），通过**多轮"Token 级复述→重训练"循环**，让训练数据从自然语言逐步演化为更适合推理的"内部语言"，同时训练翻译模型桥接回到自然语言。

## 当前状态 🚧

| 阶段 | 状态 |
|------|------|
| v0 代码构建 | ✅ 完成 (2026-04-29) |
| Windows 兼容修复 | ✅ 完成 (2026-05-01) |
| 断点续训支持 | ✅ 完成 (2026-05-01) |
| Round 0 基线训练 | 🔄 进行中 |
| 迭代训练 (Round 1-20) | ⏳ 待开始 |
| 对照实验 | ⏳ 待开始 |

详见 [`PROGRESS.md`](./PROGRESS.md)

## 核心假设

1. **自然语言不是最优推理介质**：人类语言充满歧义、冗余，理论上存在更适合 LLM 推理的"内部语言"
2. **迭代自蒸馏可演化内部语言**：通过反复 token-level 复述→重训练循环，token 分布会自发偏离自然语言
3. **翻译桥接保持可解释性**：并行训练的翻译模型保证内部语言不会退化为无意义噪声

## 技术概要

| 组件 | 实现 |
|------|------|
| 基座模型 | SmolLM2-135M-Instruct (HuggingFace) |
| 原始词表 | 49,152 |
| 内部 token | 4,096 (`<INTERNAL_0>` ~ `<INTERNAL_4095>`) |
| 特殊 token | 8（START/END/PAD/TASK 标记） |
| 总词表 | 53,256 |
| 翻译模型 | ~30M Encoder-Decoder Transformer |
| 数据集 | sonnet4.6-general/code/math/psychology (122,373 条, 763MB) |
| 硬件 | RTX 3060 12GB |
| 迭代轮数 | 20 轮 |
| 混合比例 | 80% 内部 + 20% 原始 |

## 目录结构

```
iterative-token-selfdistill/
├── README.md                    # 本文件
├── PROGRESS.md                  # 进度跟踪
├── feasibility-analysis.md      # 可行性分析
├── architecture.md              # 系统架构（设计文档）
├── experiment-plan.md           # 实验计划与里程碑
├── references.md                # 参考资料
├── train.py                     # 训练入口
├── configs/
│   └── smollm2.yaml             # 主配置文件
├── src/
│   ├── model/
│   │   └── smollm2_internal.py  # 扩展词表的 SmolLM2 模型
│   ├── tokenizer/
│   │   └── extended_tokenizer.py # 扩展分词器
│   ├── trainer/
│   │   ├── iterative_trainer.py  # 多轮迭代主控制器
│   │   ├── training_loop.py      # 单轮训练循环
│   │   └── rephrase.py           # Token 级复述生成
│   ├── translator/
│   │   └── model.py              # 内部语言→NL 翻译模型
│   ├── data/
│   │   └── dataset.py            # 数据集与 DataLoader
│   └── eval/
│       └── metrics.py            # 评估指标
├── checkpoints/                 # 模型 checkpoint（轮次 + step 级）
├── paper/                       # 论文草稿
└── sonnet4.6-*.jsonl            # 训练数据
```

## 快速开始

```bash
# 安装依赖
pip install torch transformers pyyaml tqdm

# 仅跑 Round 0 基线
python train.py --round-0-only --max-steps 500 --max-samples 5000

# 完整 20 轮训练
python train.py

# 从 checkpoint 继续
python train.py --resume checkpoints/round_5_step_3000
```

## 训练流程

```
Round 0: 基线 Fine-tuning（自然语言数据）
    ↓
Round K (K=1..20):
    1. Rephrase: Model_K 生成内部 token 序列（温度 1.2→0.8）
    2. Mix: 80% 内部数据 + 20% 原始数据
    3. Train: 联合训练 + 翻译模型训练
    4. Evaluate: 熵、token 使用率、翻译 BLEU
    5. Save: 轮次 checkpoint + step 级自动保存
```

## 关键创新

- **4096 个内部离散 token**：不是连续向量，而是离散符号，可被标准 LM 直接处理
- **多轮迭代自蒸馏**：我们无法预先决断哪种语言适合推理，只有通过演化才能发现
- **翻译模型桥接**：Encoder-Decoder Transformer 保证可解释性
- **Entropy Bonus**：防坍塌正则化，鼓励 token 多样性
- **Step 级断点保存**：每 1000 步自动存 checkpoint，防意外中断
