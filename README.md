# From Token to Agent · 从分词到智能体

> **🇨🇳 中文（默认）** · [🇬🇧 English](./README.en.md)
>
> 从分词和预训练，到 RLVR、长程智能体、自改进——
> 从零搭建现代 AI 栈。

---

## Learning Signal Ladder

```
labels              ←  W1   监督标签：把世界压成离散 token
word structure      ←  W1   词结构：BPE / Word / Char 的归纳偏置
next-token          ←  W1-W7  预测下一个 token：Transformer + Scaling Law
preferences         ←  W8-W10 SFT / DPO：人类偏好对齐
verifiers           ←  W10-W11 RLVR / ORM / PRM：可验证奖励
environment         ←  W12-W13 Coding Agent / Memory
self-judge          ←  W14-W16 自评判 / 轨迹蒸馏 / 长程自治
```

唐杰《高级机器学习》2026 秋季 · 16 周课程 · [完整大纲](./COURSE.md)

---

## 进度（截至 W5）

| 阶段 | 状态 | 里程碑 |
|---|---|---|
| HW1 Foundations | 🟡 in progress | toy-5m 跑通，正在替换为 baseline-100m |
| HW2 Systems | 🟡 in progress | Triton attention kernel + benchmark |
| HW3 Data + Scaling | ⚪ not started | W6 开工 |
| HW4 Post-Training | ⚪ not started | W8 开工 |
| Final Agent | ⚪ not started | W8 启动 proposal |

**当前进行**：W4 — compute kernels & parallelism。

详见 [ROADMAP.md](./ROADMAP.md)。

---

## 这个仓库是什么

**不是 4 份独立作业。** 是一条从 `raw text` 走到 `self-improving agent` 的不断升级的系统主线：

```
raw data
  ↓
tokenizer (W1)
  ↓
0.1B base model (W2-W3)
  ↓
systems optimization (W4-W5)
  ↓
better data + scaling (W6-W7)
  ↓
SFT / DPO / RLVR (W8-W11)
  ↓
verifiable environment (W12)
  ↓
harness + long-horizon agent (W12-W13)
  ↓
self-judge + trajectory distillation (W14-W16)
```

每一周的代码会接住上一周的产物，所以 16 周后仓库本身就是作品。

---

## 仓库结构

```
from-token-to-agent/
├── weeks/              ★ 第一主线：16 周学习节奏（W1-W5 已建）
├── assignments/        ★ 第二主线：作业 milestone 提交视图
├── src/token_to_agent/ ★ 唯一正式代码区
│   ├── tokenizer/      BPE 实现（HW1）
│   ├── model/          decoder-only Transformer（HW1）
│   ├── training/       AdamW + cosine LR（HW1）
│   ├── kernels/        Triton + reference attention（HW2）
│   └── (serving/ data/ post_training/ rl/ environments/ harness/ memory/ evaluation/ utils/)
│                       待对应周开工时填充
├── tests/              按模块划分
├── configs/            训练 / 系统 / 后训练 / agent 配置
├── experiments/        每实验 5 件套：config + metadata + metrics + README
├── extensions/         主线之外的增量实验（EXT-W<N>-<idx>）
├── docs/               课程来源 / 架构图 / 论文阅读笔记
├── artifacts/          本地产物（gitignored）
├── scripts/            5 个 CLI：train / evaluate / generate / benchmark / serve
├── COURSE.md           16 周完整大纲（冻结）
└── ROADMAP.md          进度快照
```

---

## 快速开始

```bash
# 安装
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 跑测试
bash tests/test.sh

# 训练 toy 0.1B
bash scripts/setup_env.sh
python scripts/train.py --config configs/pretrain/toy-5m.yaml

# 推理
python scripts/generate.py --checkpoint artifacts/checkpoints/toy-5m --prompt "你好"

# benchmark（W2）
python scripts/benchmark.py attention --backend triton --seq-len 128 1024
```

> ⚠️ 本仓库代码**完整可跑**，但 0.1B 规模实验需要在带 GPU 的机器上执行。
> Jetson Nano 仅作开发 / 教学环境，toy-5m 在 CPU 上约 5 分钟跑通。

---

## 16 周节奏

| 时间 | 交付 | 状态 |
|---|---|---|
| W1-W3 | HW1 Foundations | 🟡 in progress |
| W4-W5 | HW2 Systems | 🟡 in progress |
| W6-W7 | HW3 Data + Scaling | ⚪ not started |
| W8 | Final Project Proposal | ⚪ not started |
| W9-W11 | HW4 Post-Training | ⚪ not started |
| W10 | Final Mid-Report | ⚪ not started |
| W12-W15 | Agent / Harness / Verifier | ⚪ not started |
| W16 | Paper + Live Demo | ⚪ not started |

每个阶段打 Git tag：`v0.1-hw1-foundations` / `v0.2-hw2-systems` / … / `v1.0-from-token-to-agent`。

---

## License

Apache 2.0 — 见 [LICENSE](./LICENSE)。

---

## 致谢

- 课程来源：唐杰《高级机器学习》2026 秋季 · [PPT 摘录](./docs/course-origin/)
- 课程参考：HKUDS/CLI-Anything（agent-native CLI 设计方法论）
- 论文参考：THUDM GLM 系列（GLM-4.5 / GLM-5 / slime / DeepDive / ReST-MCTS 等）