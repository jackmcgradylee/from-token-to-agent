# From Token to Agent · 从 Token 到智能体

> **🇨🇳 中文（默认）** · [🇬🇧 English](./README.en.md)
>
> 从 Tokenizer、预训练与 Scaling，到 RLVR、长程 Agent 与 Self-Evolution。
> **亲手走通现代大模型从 Token 到 Agent 的完整链路。**

---

## 课程来源

本仓库基于 **唐杰教授 2026-09-17 在清华大学《高级机器学习》课程中公布的 16 周课程地图与作业要求**，进行个人复现、实验与扩展。

目标不是把课程压缩成几个 Demo，而是在可承受的计算规模下，尽可能完整地走一遍：

- Tokenization 与 Transformer
- Pretraining 与 Scaling
- GPU Kernel、并行训练与推理系统
- Pretraining Data 与 Synthetic Data
- SFT、Preference Learning 与 RLVR
- Long-Horizon Agent RL
- Harness、Memory 与 Continual Learning
- Self-Evaluation、Evaluation 与 Safety

课程 PPT 给出了一条贯穿 16 周的 Learning Signal Ladder：

![课程地图（6 张拼图：LLM 用例金字塔 / Scaling↔Generalization 时间线 / 课程导言"让机器像人一样思考" / 2026 研究优先级 / 16 周课程地图 / 历史脉络——三次范式转移，1980s→2026）](./docs/course-origin/tang-2026-09-17-aml-ppt-collage.jpg)

完整课程计划、每周实验与验收标准见 [`COURSE.md`](./COURSE.md)。

---

## Learning Signal Ladder

```text
labels
↓
word structure
↓
next-token
↓
preferences
↓
verifiers
↓
environment
↓
self-judge
```

我把这条线理解为课程真正的主轴：

> 学习信号从人工标注，逐渐转向数据自身、Preference、程序化 Verifier、Environment Feedback 和 Self-Evaluation；信号越来越可规模化，人也逐步退出反馈闭环。

---

## 16 周 Course Map

```
W1  Three Paradigm Shifts
W2  Architecture Revisited
W3  Training Dynamics & Scaling Laws
W4  Compute, Kernels, Parallelism
W5  Economics of Inference
W6  Pretraining Data
W7  Synthetic Data & Governance
W8  SFT & Distillation
W9  Preference Learning
W10 RLVR & Reasoning
W11 Agent RL: Long-Horizon
W12 Agent Foundations
W13 Memory & Continual Learning
W14 Self-Evaluation & Evolution
W15 Evaluation & Safety
W16 What Comes After LLMs?
```

每周 Topic、实验与验收标准见 [`COURSE.md`](./COURSE.md)。

---

## 5 个阶段性 Deliverables

> 16 周 Topic 定义**学什么**，5 个正式作业定义**做出什么**。两者相互关联，但并不是严格的一周一作业映射——尤其 Final Project 从 W8 起就与 HW4 并行推进。

| 作业 | 主要覆盖内容 | 核心交付 |
| --- | --- | --- |
| **HW1 Foundations** | Tokenization / Architecture / Training | Tokenizer + Transformer + ~0.1B Pretraining + Ablation |
| **HW2 Systems** | Kernel / Parallelism / Inference | Triton Kernel + Multi-GPU + Serving + Cost |
| **HW3 Data + Scaling** | Pretraining Data / Synthetic Data / Scaling Law | Raw Pipeline + Data Card + Scaling Extrapolation |
| **HW4 Post-Training** | SFT / Preference / RLVR / Agent RL | Controlled Comparison（同 Base / 同 Eval / 同成本） |
| **Final Project** | W8 起并行贯穿后半程 | Verifiable Env + Harness + Long-Horizon Agent + Self-Judge |

每个作业的 milestone 视图见 [`assignments/`](./assignments/)。

---

## 当前进度

**Current: W2 — Architecture Revisited (RMSNorm + GQA ablations done)**

| 作业 | 状态 |
| --- | --- |
| **HW1 Foundations** | 🟢 完成（W1 tokenizers + EXT-W1-01 BPE vocab sweep） |
| **HW2 Systems** | 🟠 实现完成，ablation 已交付（W2 RMSNorm/GQA + W4 Triton 起步） |
| **HW3 Data + Scaling** | ⚪ 未开始（W6） |
| **HW4 Post-Training** | ⚪ 未开始（W8） |
| **Final Project** | ⚪ 未开始（W8 启动 proposal） |

**最近进展**

- ✅ **EXT-W1-01** — BPE vocab sweep: 在 21 KB toy corpus 上饱和于 `actual_vocab=530`，target ≥ 1024 时 trainer 提前终止，bytes/token 冻结在 1.214。
- ✅ **W2 exp-001/002** — RMSNorm vs LayerNorm（参数差 +640）+ MHA vs GQA（KV cache 4× shrink），59/59 tests PASS。
- 🔜 **W3** — Training dynamics + scaling law hold-out 预测实验。

详细实验、扩展与下一步计划见 [`ROADMAP.md`](./ROADMAP.md)。

---

## 仓库结构

```
from-token-to-agent/
├── weeks/         16 周学习与实验记录
├── assignments/   5 个正式作业与 Final 提交
├── src/           持续演化的统一实现（token_to_agent/ 命名空间）
├── tests/         correctness tests
├── configs/       可复现实验配置
├── experiments/   实验记录与结果
├── extensions/    课程之外的研究实验（EXT-W<N>-<idx>）
├── docs/          课程来源 / 架构图 / 论文阅读
├── artifacts/     本地产物（gitignored）
├── scripts/       CLI 入口
├── COURSE.md      16 周完整课程计划
└── ROADMAP.md     当前进度
```

`src/` 内部详细子包划分见 `src/token_to_agent/__init__.py` 顶部 docstring 与 `COURSE.md §23`。

---

## Quick Start

> 已完成模块均提供可复现实验入口；后续模块按 16 周计划持续建设。

```bash
# 安装
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 跑测试
bash tests/test.sh

# 训练链路快速验证（约 5M，仅做 smoke test）
python scripts/train.py --config configs/pretrain/toy-5m.yaml

# HW1 正式 baseline（约 100M，需要 GPU）
python scripts/train.py --config configs/pretrain/baseline-100m.yaml

# 用 checkpoint 生成
python scripts/generate.py --checkpoint artifacts/checkpoints/toy-5m --prompt "你好"

# Attention kernel benchmark（W4 / HW2）
python scripts/benchmark.py attention --backend pytorch triton --seq-len 128 1024
```

> Jetson Nano / CPU 环境仅适合开发与 smoke test；正式 baseline 实验需要在带 GPU 的机器上运行。

---

## License

Apache 2.0 — 见 [LICENSE](./LICENSE)。

---

## 致谢

- **课程来源**：唐杰《高级机器学习》2026-09-17 · [PPT 摘录](./docs/course-origin/tang-2026-09-17-aml-ppt-collage.jpg)
- **研究坐标**：THUDM GLM 系列（GLM-4.5 / GLM-5 / slime / DeepDive / ReST-MCTS / TDRM / AgentTuning / AgentBench / SCALE-CUA / INFTY）
- **延伸阅读**：[`docs/reading/`](./docs/reading/)