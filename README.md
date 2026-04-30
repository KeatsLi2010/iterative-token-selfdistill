# 迭代式 Token 级自蒸馏语言演化实验

## 一句话摘要

训练 135M 小模型，通过**多轮"Token 级复述→重训练"循环**，让训练数据从自然语言逐步演化为某种"更适合推理的内部语言"，同时训练一个翻译模型将内部语言映射回自然语言。

## 核心假设

1. **自然语言不是最优推理介质**：人类语言充满歧义、冗余和信息熵不足，理论上存在更适合 LLM 推理的"内部语言"
2. **迭代自蒸馏可演化内部语言**：通过反复的 token-level 复述→重训练循环，token 分布会自发偏离自然语言，收敛到一种信息密度更高、更"推理友好"的表征
3. **翻译桥接保持可解释性**：并行训练的翻译模型保证内部语言不会退化为无意义的噪声，始终能映射回自然语言

## 目录结构

```
iterative-token-selfdistill/
├── README.md                    # 本文件
├── feasibility-analysis.md      # 详细可行性分析
├── architecture.md              # 系统架构设计
├── experiment-plan.md           # 实验计划与里程碑
├── src/                         # 源代码（待实现）
│   ├── model/                   # 135M 模型定义
│   ├── trainer/                 # 迭代训练循环
│   ├── translator/              # 翻译模型
│   ├── eval/                    # 评估模块
│   └── data/                    # 数据处理
├── configs/                     # 配置文件
├── experiments/                 # 实验记录
└── results/                     # 结果输出
```

## 快速导航

- 想了解可行性？→ [`feasibility-analysis.md`](./feasibility-analysis.md)
- 想看技术架构？→ [`architecture.md`](./architecture.md)
- 想看实验计划？→ [`experiment-plan.md`](./experiment-plan.md)
