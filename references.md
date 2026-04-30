# 相关论文与项目索引

> 搜索时间：2026-04-29 | 搜索范围：arXiv, GitHub

---

## 一、核心相关（高优先级必读）

### 1. ⭐ Large Language Model as Token Compressor and Decompressor
- **arXiv**: [2603.25340](https://arxiv.org/abs/2603.25340)
- **作者**: Wenbing Li, Zikai Song, Jielei Zhang 等
- **时间**: 2026 年 3 月
- **核心思路**: 用预训练 LLM 将长文本压缩为离散的 **Z-tokens**（内部语言），然后解压回原文。自适应压缩：语义密集处分配更多 Z-tokens，冗余处激进压缩。使用 LoRA adapter 实现。
- **结果**: Wikipedia 上达到 18× token 压缩率，保持重建保真度和下游任务性能
- **与本项目的异同**:
  - ✅ 相同：内部离散 token 表征、压缩→解压双向、LoRA 高效训练
  - ❌ 不同：不迭代、不做推理优化、不做语言演化、用预训练 LLM 而非从头训练
- **价值**: **最相似的近期工作**，可作为技术基础，你的方案在其上添加"迭代演化"即可差异化

### 2. ⭐ SPIN: Self-Play Fine-Tuning Converts Weak LMs to Strong LMs
- **arXiv**: [2401.01335](https://arxiv.org/abs/2401.01335)
- **作者**: Zixiang Chen, Yihe Deng, Huizhuo Yuan 等 (UCLA)
- **时间**: ICML 2024
- **核心思路**: Self-play 机制——LLM 与自己的历史版本对弈，生成训练数据，逐步从弱模型演化到强模型。理论上证明全局最优解仅在 LLM 策略与目标数据分布对齐时达到
- **代码**: https://github.com/uclaml/SPIN
- **价值**: 迭代自训练的 SOTA 方法 + 开源代码，直接可参考训练循环

### 3. ⭐ Gisting: Learning to Compress Prompts with Gist Tokens
- **arXiv**: [2304.08467](https://arxiv.org/abs/2304.08467)
- **作者**: Jesse Mu, Xiang Lisa Li, Noah Goodman (Stanford)
- **时间**: NeurIPS 2023
- **核心思路**: 训练 LM 将 prompt 压缩为少量 "gist tokens"，可缓存复用。通过修改 Attention mask 实现
- **结果**: 26× prompt 压缩，40% FLOPs 降低
- **价值**: Gist token 的工程实现参考，Attention mask 技巧

### 4. ⭐ CoCoMix: LLM Pretraining with Continuous Concepts
- **arXiv**: [2502.08524](https://arxiv.org/abs/2502.08524)
- **作者**: Jihoon Tack, Jack Lanchantin 等 (Meta FAIR)
- **时间**: 2025 年 2 月
- **核心思路**: 在标准 next token prediction 中混合连续概念预测。概念通过预训练的 sparse autoencoder 学习，插入到 hidden state 中与 token 交替
- **结果**: 在语言建模和下游推理任务上一致优于标准 NTP、知识蒸馏和 pause tokens
- **价值**: 证明"非 token 级内部表征"有助于推理——支持你的核心假设

---

## 二、重要相关（中优先级）

### 5. Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
- **arXiv**: [2403.xxxxx](https://arxiv.org/abs/2403.xxxxx) (需确认)
- **作者**: Eric Zelikman, Georges Harik 等 (Stanford)
- **时间**: 2024 年 3 月
- **核心思路**: 在每步 token 预测前插入"思考 token"，模型学习在内部进行推理
- **价值**: "内部推理 token"的思路，与你的内部语言概念相通

### 6. Born Again Neural Networks
- **arXiv**: [1805.04770](https://arxiv.org/abs/1805.04770)
- **作者**: Tommaso Furlanello, Zachary Lipton 等
- **时间**: ICML 2018
- **核心思路**: 知识蒸馏中，学生模型（相同容量）可以超过教师模型，通过多代蒸馏迭代提升
- **价值**: 迭代自蒸馏有效性的经典理论支撑

### 7. SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning
- **时间**: 2026 年 3 月
- **核心思路**: 在 span 级别插入 pause tokens 进行隐式推理，比逐 token pause 更高效
- **价值**: Token 级内部推理的最新进展

### 8. How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse
- **arXiv**: [2404.xxxxx](https://arxiv.org/abs/2404.xxxxx)
- **作者**: Mohamed El Amine Seddik 等
- **时间**: 2024 年 4 月
- **核心思路**: 对"模型坍缩"现象的统计分析——迭代使用合成数据会逐步丢失尾部分布
- **价值**: ⚠️ 直接警示你的迭代方案的风险，必读

### 9. Models of Symbol Emergence in Communication
- **时间**: 2023 年 3 月
- **作者**: Julian Zubek, Tomasz Korbak 等
- **核心思路**: 综述通信中符号涌现的计算模型，避免局部最优
- **价值**: Emergent Communication 的理论指导

---

## 三、背景相关（低优先级，扩展阅读）

### 10. Characterizing Model Behavior Under Synthetic Data Training
- **arXiv**: [2510.05133](https://arxiv.org/abs/2510.05133)
- **时间**: 2025 年 10 月
- **核心思路**: 跨规模和混合比例的合成数据训练实证研究
- **价值**: 数据混合比例的经验指导

### 11. Discrete VAE / dVAE 系列
- **代表工作**: DALL-E 的 dVAE (Ramesh et al., 2021), VQ-VAE (van den Oord et al., 2017)
- **核心思路**: 学习离散隐变量表示，通过 codebook 量化
- **价值**: 离散隐变量学习的技术参考

---

## 四、开源项目

### SPIN (Self-Play Fine-Tuning)
- **GitHub**: https://github.com/uclaml/SPIN
- **语言**: Python / PyTorch
- **相关度**: ⭐⭐⭐⭐⭐ (迭代自训练框架，可直接改造)

### HuggingFace TRL (Transformer Reinforcement Learning)
- **GitHub**: https://github.com/huggingface/trl
- **相关度**: ⭐⭐⭐ (自训练 + PPO/DPO 工具链)

### nanoGPT
- **GitHub**: https://github.com/karpathy/nanoGPT
- **相关度**: ⭐⭐⭐⭐ (135M 级模型的直接参考实现)

### LLM as Token Compressor (暂无公开代码)
- 论文 2026 年 3 月刚发布，代码可能尚未开源
- **建议**: 关注作者 GitHub 或联系获取

---

## 五、你的方案的独特定位

| 维度 | Gisting | Z-Token | SPIN | CoCoMix | **本方案** |
|------|---------|---------|------|---------|-----------|
| 内部表征 | Gist tokens | Z-tokens | 自然语言 | 连续概念 | **Internal tokens** |
| 是否迭代 | ❌ | ❌ | ✅ | ❌ | **✅ 多轮** |
| 是否翻译回 NL | ❌ | ✅ | ❌ | ❌ | **✅ 翻译模型** |
| 推理优化 | ❌ | ❌ | ✅ | ✅ | **✅ 下游任务损失** |
| 从零训练 | ❌ | ❌ | ❌ | ✅ | **✅ 135M 从头** |
| 语言演化 | ❌ | ❌ | ❌ | ❌ | **✅ 核心目标** |

**差异化价值**: 你的方案是目前唯一同时包含「内部离散 token + 多轮迭代 + 翻译桥接 + 推理驱动」的组合。即使最终不 work，negative result 也有发表价值。
