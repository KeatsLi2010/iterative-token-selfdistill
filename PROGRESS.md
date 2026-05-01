# PROGRESS.md — 项目进度

> 最后更新: 2026-05-01

## 总体进度

```
████████░░░░░░░░░░░░░░░░░░ ~30% — v0 代码完成，Windows 适配中，基线训练进行中
```

## 里程碑

### ✅ M0: 初始代码构建 (2026-04-29)
- [x] 项目框架搭建
- [x] SmolLM2-135M-Instruct 加载 + 词表扩展
- [x] ExtendedTokenizer（53,256 词表）
- [x] ChatDataset / InternalSequenceDataset
- [x] SingleRoundTrainer（带 entropy bonus）
- [x] RephraseGenerator
- [x] InternalTranslator (~30M params)
- [x] 评估指标（entropy、token 使用率、压缩率）
- [x] YAML 配置 + CLI 参数
- [x] train.py 入口脚本

### ✅ M1: Windows 兼容修复 (2026-05-01)
- [x] `collate_fn` lambda → `functools.partial`（Windows pickle 问题）`bf982ed`
- [x] `torch.compile` → Triton 不可用时自动跳过 `0ef5a7e`
- [x] `ZeroDivisionError` in LR scheduler `8c878aa`
- [x] 本地模型路径 + HF 镜像回退 `f1c5737`
- [x] 自动断点保存（step 级 + round 级）`83bc77d`
- [x] GitHub 认证配置

### 🔄 M2: Round 0 基线训练 (进行中)
- [ ] 自然语言数据 fine-tune
- [ ] 验证基座模型 on RTX 3060
- [ ] 记录基线 perplexity / 下游任务指标

### ⏳ M3: 迭代训练 (Round 1-5, 短期验证)
- [ ] Round 1 复述 + 训练 + 评估
- [ ] 内部 token 使用率是否增长？
- [ ] 翻译 BLEU 是否保持？
- [ ] 前 5 轮趋势分析
- [ ] 决定是否继续完整 20 轮

### ⏳ M4: 完整 20 轮迭代
- [ ] 运行全部 20 轮
- [ ] 每轮记录：entropy、token 使用率、压缩率、翻译 BLEU
- [ ] 生成可视化报告

### ⏳ M5: 对照实验
- [ ] No-Internal（自然语言复述替代）
- [ ] No-Task（去掉推理损失）
- [ ] No-Iter（仅 1 轮）
- [ ] Random-Internal（下界 baseline）

### ⏳ M6: 论文撰写
- [ ] 实验数据分析
- [ ] LaTeX 论文

## 已知问题

| 问题 | 状态 | 备注 |
|------|------|------|
| `pin_memory=True` 无 CUDA 警告 | 🟡 低优 | 设 `num_workers=0` 时不影响 |
| Triton 不支持 Windows | ✅ 已修复 | `torch.compile` 自动跳过 |
| 训练速度待优化 | 🟡 观察中 | Round 0 跑通后评估 |

## 配置快照

- **模型**: SmolLM2-135M-Instruct
- **GPU**: RTX 3060 12GB
- **Batch size**: 32
- **Max steps/round**: 5000
- **总轮数**: 20
- **数据**: 122,373 条 chat 格式 JSONL
- **保存频率**: 每 1000 步
