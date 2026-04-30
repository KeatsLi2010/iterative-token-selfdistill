# 实验计划与里程碑

## 时间线总览（预计 6-8 周）

```
Week 1-2    ████████  阶段 0：环境搭建 + 基线模型
Week 2-3    ████████  阶段 1：Token 体系实现 + 单轮验证
Week 3-5    ████████████  阶段 2：迭代循环实现 + 消融实验
Week 5-7    ████████████  阶段 3：完整实验运行 + 分析
Week 7-8    ████████  阶段 4：论文撰写 + 清理
```

---

## 阶段 0：环境搭建与基线（第 1-2 周）

### 0.1 环境准备
- [ ] 搭建 PyTorch 2.x + CUDA 12.x 环境
- [ ] 克隆/创建 GPT-2 small 实现
- [ ] 准备训练数据（WikiText-103 或 C4 子集，~100M tokens）
- [ ] 准备下游评估数据（GSM8K, MBPP, LogiQA 子集）
- [ ] 配置 WandB/MLflow

### 0.2 基线模型
- [ ] 训练标准 135M LMs 的基线（仅在自然语言上训练）
- [ ] 记录 baseline 的 PPL、下游任务准确率
- [ ] 作为所有后续实验的对比基线

### 交付物
- `src/model/gpt2_small.py` — 模型定义
- `src/data/dataset.py` — 数据加载
- `configs/baseline.yaml` — 基线配置
- `results/baseline/metrics.json` — 基线指标

---

## 阶段 1：Token 体系 + 单轮验证（第 2-3 周）

### 1.1 Tokenizer 扩展
- [ ] 扩展 GPT-2 tokenizer，添加 4096 个 internal tokens + 特殊 tokens
- [ ] 实现 token masking 逻辑（限制采样范围）
- [ ] 测试：确保 internal token 区域不会输出自然语言 token

### 1.2 单轮复述实现
- [ ] 实现复述循环：输入原文 → 生成 internal token 序列
- [ ] 实现 温度采样 + top_p + entropy bonus
- [ ] 可视化：观察 internal token 的初始分布

### 1.3 翻译模型骨架
- [ ] 实现 encoder-decoder 翻译模型
- [ ] 实现回译训练循环
- [ ] 验证翻译模型能在简单场景下工作

### 1.4 单轮完整流程
- [ ] 端到端运行：训练 → 复述 → 重训练（1 轮）
- [ ] 记录 token 熵变化、下游任务变化

### 交付物
- `src/tokenizer/extended_tokenizer.py`
- `src/trainer/rephrase.py` — 复述循环
- `src/translator/model.py` — 翻译模型
- `results/phase1/single_round_report.md`

### 里程碑检查点 🔴
> **关键判定**：单轮后，模型使用 internal token 能否在保持一定翻译保真度的同时，不显著劣于基线？如果翻译保真度 < 20 BLEU，调整方案。

---

## 阶段 2：迭代循环 + 消融实验（第 3-5 周）

### 2.1 迭代训练循环
- [ ] 实现完整的 K 轮迭代控制器
- [ ] 实现数据混合（20% 原始 + 80% 内部）
- [ ] 实现 checkpoint 管理和恢复

### 2.2 推理任务损失集成
- [ ] 将下游任务（数学/代码/逻辑）作为训练目标加入
- [ ] 实现 multi-task 训练（LM + 下游任务交替 batch）
- [ ] 任务 token (`<TASK_MATH>`) 的条件控制

### 2.3 防坍塌机制
- [ ] Entropy bonus 损失实现
- [ ] Token 使用率监控
- [ ] 自适应温度调整

### 2.4 消融实验
- [ ] **Exp-A**: Full 方案（10 轮）
- [ ] **Exp-B**: 无 Internal Token（自然语言复述对照）
- [ ] **Exp-C**: 无推理任务损失（纯重建）
- [ ] **Exp-D**: 无迭代（单轮）
- [ ] **Exp-E**: 随机 Internal Token（下界 baseline）

### 交付物
- `src/trainer/iterative_trainer.py`
- `src/eval/metrics.py` — 评估指标
- `configs/experiments/` — 各实验配置
- `results/phase2/ablation_report.md`

### 里程碑检查点 🔴
> **关键判定**：
> 1. 5 轮后，Exp-A 的下游任务准确率是否优于 Exp-D（单轮）？
> 2. Exp-A 的有效 token 使用率是否保持在 50% 以上（未坍塌）？
> 3. Exp-B（自然语言复述）vs Exp-A（内部 token）对比 → internal token 是否有优势？
>
> 如果三项全否，需重新评估方案可行性。

---

## 阶段 3：完整实验与分析（第 5-7 周）

### 3.1 完整 20 轮运行
- [ ] 最优配置下运行完整 20 轮迭代
- [ ] 每轮保存所有指标和 checkpoint
- [ ] 记录 token 分布演化视频/动图

### 3.2 内部语言分析
- [ ] Token 共现矩阵分析（internal tokens 是否形成结构化关系）
- [ ] 聚类分析（是否存在语义聚类）
- [ ] 信息论分析（互信息、压缩率）
- [ ] 与自然语言 token 的对应关系分析

### 3.3 翻译质量深度评估
- [ ] 人类评估（小规模，50 条样本）
- [ ] 案例分析：成功的翻译 vs 失败的翻译
- [ ] 翻译保真度随轮数的变化趋势

### 3.4 推理能力深度分析
- [ ] 逐任务分析（数学 vs 代码 vs 逻辑，哪个受益最多？）
- [ ] 注意力模式分析（内部 token 的注意力分布）
- [ ] 与基线模型的逐样本对比

### 交付物
- `results/phase3/full_20_rounds/` — 完整结果
- `results/phase3/internal_language_analysis.md`
- `results/phase3/translation_deep_dive.md`
- `results/phase3/reasoning_analysis.md`
- 关键图表（token 熵曲线、下游任务曲线、翻译 BLEU 曲线）

### 里程碑检查点 🔴
> **核心判定**：
> 20 轮后，Exp-A 的下游任务是否有 > 5% 的相对提升？
> 如果没有 → 方案可能不 work，转为写 negative result 论文。

---

## 阶段 4：总结与论文（第 7-8 周）

### 4.1 论文大纲
```
1. Introduction — 自然语言不是最优推理介质
2. Related Work — Self-Distillation, Emergent Communication, DVAE
3. Method — Iterative Token-Level Self-Distillation
4. Experiments — 消融实验 + 分析
5. Analysis — 内部语言的结构
6. Discussion — 局限性、未来工作
7. Conclusion
```

### 4.2 代码整理
- [ ] README + 复现指南
- [ ] 依赖清单 (requirements.txt / poetry.lock)
- [ ] 预训练 checkpoint 上传（HuggingFace Hub）

### 交付物
- `paper/` — LaTeX 源码
- `README.md` — 项目文档
- 开源发布（GitHub + HuggingFace）

---

## 风险评估与应急预案

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Token 多样性在 5 轮内崩溃 | 高 (40%) | 实验无效 | 加强防坍塌机制，降低每轮学习率 |
| 翻译质量过低导致不可解释 | 中 (25%) | 论文说服力下降 | 提高翻译模型容量，或改为软翻译（embedding 对齐） |
| 下游任务无显著提升 | 高 (50%) | 核心假设被推翻 | 转为 negative result 论文，或改为有监督优化（强化学习） |
| 计算资源不足 | 低 (10%) | 进度延迟 | 降低轮数到 10 轮，减少 batch size |
| 内部语言完全退化为压缩编码 | 中 (30%) | 与目标偏离 | 接受这个结果，重新 framing 为"学习最优压缩表示" |

---

## 日常实验记录模板

```
### Round K | Date: YYYY-MM-DD

**Training:**
- Steps: XXX
- Train loss (L_rec/L_task/L_div/L_total): 0.XX / 0.XX / 0.XX / 0.XX
- Time: X hours

**Evaluation:**
- Downstream (Math/Code/Logic): XX% / XX% / XX%
- Translation BLEU: XX.X
- Token entropy: X.XX
- Effective token usage: XX%

**Observations:**
- [ ] Token 分布是否有异常？
- [ ] 翻译质量变化？
- [ ] 需要调参吗？

**Decisions for next round:**
- 
```
