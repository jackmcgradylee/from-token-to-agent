# From Token to Agent · 从分词到智能体

> **从原始文本到自改进智能体** — 亲手搭建现代大模型栈的每一级：分词、训练、系统优化、scaling、后训练（SFT/DPO/RLVR）、可验证环境、长程智能体、自评判。

> 📌 **缘起（先看这里）**：本仓库是 **唐杰老师 2026 年秋季《高级机器学习》（清华大学）** 课正式布置的 5 级 LLM ladder 大作业的**作业仓库**。作业发布日：**2026-09-17**。详见 [项目缘起](#项目缘起--2026-09-17-清华高级机器学习) 段：完整背景、PPT、评分规则。

我们的 ladder：

```
原始语料
   ↓
Tokenizer
   ↓
0.1B Base Model（Transformer 自己实现）
   ↓
Systems Optimization（Triton / KV cache / 并行）
   ↓
Better Data / Scaling Law
   ↓
SFT / DPO / RLVR
   ↓
Verifiable Environment
   ↓
Harness（智能体调用框架）
   ↓
Long-Horizon Agent
   ↓
Self-Judge
```

每一级都是下一级的输入。`src/` 是不断生长的代码库；`assignments/` 是作业交付视图；`extensions/` 是后加的研究延伸。**不新建 repo，不复制代码。**

英文版 README 见 [README.md](./README.md)。COURSE 排程与 HW/EXT 编号规范见 [COURSE.md](./COURSE.md)。

---

## 五大正式作业

| # | 主题 | 核心任务 | 必交结果 | 给下一级留的接口 |
|---|---|---|---|---|
| HW1 | Foundations — 从零搭 LM | Tokenizer + Transformer，端到端训练约 0.1B 模型 | tokenizer、模型代码、训练脚本、checkpoint、loss 曲线、生成样例、实验报告 | `src/tokenizer/`、`src/model/`、`src/training/`、`checkpoints/hw1/baseline-100m/` |
| HW2 | Systems — 把它跑快 | 手写 Triton Attention Kernel，并测试单卡/多卡训练和推理 | correctness、latency、throughput、显存、scaling efficiency | `src/kernels/`（替换 `src/model/attention.py` 的 attention 路径） |
| HW3 | Data + Scaling — 让数据可预测 | 从原始 dump 构建语料；训练不同规模模型；拟合 Scaling Law 并外推 | data pipeline、Data Card、模型实验点、Scaling 曲线、预测误差 | `src/data/`、tokenizer 重训、scaling 外推 |
| HW4 | Post-Training — SFT vs DPO vs RLVR | 从同一个 Base Model 出发，做三条后训练路线的 controlled comparison | 三套 checkpoint、统一评测、训练曲线、效果/成本比较 | `src/post_training/`，base model 来自 HW3 |
| FINAL | Agents — Environment + Harness + Long-Horizon Agent | 搭建可验证环境和 harness，训练/优化长程 Agent；加入 self-judge | Proposal、Mid Report、Agent、Harness、Verifier、最终论文、Live Demo | `src/environments/`、`src/harness/`、`src/evaluation/` |

**不要一开始就加 HW5、HW6。** 新想法全部作为 Extension 挂在五阶段下面（`EXT-101..505`）。

---

## 仓库目录

```
from-token-to-agent/
├── README.md           ← 英文版（README.md）
├── README.zh-CN.md     ← 你正在看的中文版
├── COURSE.md           ← 16 周节奏 + HW/EXT 编号规范 + 研究模板
│
├── assignments/        ← 作业交付视图（每次作业按 Problem → Motivation → Method → Results → Reproduction 模板）
│   ├── hw1-foundations/
│   ├── hw2-systems/
│   ├── hw3-data-scaling/
│   ├── hw4-post-training/
│   └── final-agent/
│
├── src/                ← 持续生长的代码库（每级作业往这里写，不复制）
│   ├── tokenizer/      ← BPE（HW1；HW3 重训）
│   ├── model/          ← Transformer（HW1；HW2 换 attention）
│   ├── training/       ← 训练循环（HW1；之后每级都复用）
│   ├── kernels/        ← Triton attention 等（HW2）
│   ├── data/           ← 语料 pipeline（HW3）
│   ├── post_training/  ← SFT/DPO/RLVR（HW4）
│   ├── environments/   ← 可验证环境（FINAL）
│   ├── harness/        ← 智能体调用框架（FINAL）
│   └── evaluation/     ← 统一评测（HW4 + FINAL）
│
├── configs/            ← 每级作业的 YAML 配置
├── experiments/        ← 实验产物
├── checkpoints/        ← 模型权重（git 忽略，按 tag 留 sidecar）
├── extensions/         ← EXT-XXX 增量工作
├── reports/            ← 跨阶段的分析
├── paper/              ← Proposal / Mid Report / Final Paper
├── scripts/            ← CLI 入口
├── data/               ← 原始 + 处理后数据（git 忽略）
└── tests/              ← 单测
```

### 为什么这样切分？

`assignments/` 是**交作业视图**——按研究模板写短文档。
`src/` 是**持续生长的代码**。HW2 不是把 HW1 的 Transformer 复制一份，而是在同一个 `src/model/attention.py` 旁边放一个 `src/kernels/attention.py`，让模型自己选；HW3 不是新 tokenizer，而是重训现有的那一个。**代码积累，不复制。**

---

## 里程碑 & Git Tag

| 周 | 交付 | Git tag |
|---|---|---|
| W1–W3 | HW1 Foundations | `v0.1-hw1-foundations` |
| W4–W5 | HW2 Systems | `v0.2-hw2-systems` |
| W6–W7 | HW3 Data + Scaling | `v0.3-hw3-scaling` |
| W8 | Final Project Proposal | `v0.4-proposal` |
| W9–W11 | HW4 Post-Training | `v0.5-post-training` |
| W10 | Final Mid Report | `v0.6-midthesis` |
| W12–W15 | Agent / Harness / Verifier | `v0.7-agent` |
| W16 | 论文 + Live Demo | `v1.0-final` |

GitHub 时间线因此变成：

```
v0.1 → v0.2 → v0.3 → v0.4 → v0.5 → v0.6 → v0.7 → v1.0
```

每个 tag 都是"那一级的完整 ladder 已经能工作"的快照，不是"只有那一级的代码合并了"。

---

## 一句话定位

> **从分词到智能体**

不是营销。每一行字在仓库里都有对应代码。

---

## 快速开始

```bash
# 装环境（脚本会建 venv + 装 torch CPU 等）
bash scripts/setup_env.sh

# 一键重跑 HW1 toy-5m（5 分钟在 Jetson CPU 上能跑完）
.venv/bin/python scripts/make_sample_corpus.py --out data/raw/sample.txt --size-mb 1
.venv/bin/python scripts/train.py --config configs/hw1/toy-5m.yaml
.venv/bin/python scripts/generate.py \
    --ckpt checkpoints/hw1/toy-5m/model.pt \
    --tokenizer checkpoints/hw1/toy-5m/tokenizer.json \
    --prompts-file assignments/hw1-foundations/results/prompts.txt \
    --out assignments/hw1-foundations/results/samples.txt

# 跑 baseline-100m（需要 GPU 机器）
.venv/bin/python scripts/train.py --config configs/hw1/baseline-100m.yaml --device cuda
```

完整 reproduction 见 `assignments/hw1-foundations/README.md`。

---

## 当前状态

- ✅ **v0.1-hw1-foundations**（2026-09-17）：tokenizer + Transformer + 训练循环端到端跑通。Jetson CPU 上 toy-5m 验证：2.5M 参数，训练 6.43 → 2.27，val 2.26。Baseline-100m 配置已 committed，等 GPU。
- 🚧 **HW2 Systems**：Triton attention kernel + benchmark。下一站。

## 边界（这个仓库不做什么）

- **不是五个孤立的课程项目**。代码积累。
- **不是在 0.1B 模型上卷 SOTA**。HW1-3 是**理解**每一级，不是打榜。真正值得长期投入研究的地方，从 HW4 的 Verifier/RLVR 开始，到 Final 的 Environment/Harness/Evaluation/Self-Improvement。
- **不是第一天就搭"什么都会的通用 Agent 框架"**。Final 第一版 = 一个可验证环境 + 一个 harness + 一个 agent。泛化走 Extension 路线。

---

## 致谢与缘起

课程命题来自唐杰老师大作业的 *"A Ladder That Climbs the Course"* ——
"前一个作业产出的东西，成为后一个作业的输入"。

仓库本身的研究模板（Problem → Motivation → Method → Results → Reproduction）来自那张 PPT 最底部那一行。

---

## 项目缘起 — 2026-09-17 清华《高级机器学习》

本仓库**直接是这门课的大作业**，不是独立项目。每个决策——5 级 ladder、0.1B baseline、Triton attention 必做、scaling law hold-out 预测、SFT/DPO/RLVR 同基座对照、可验证环境 + harness + 长程 agent + self-judge 收尾——都来自一个唯一来源：

> **唐杰老师 — 清华大学 2026 年秋季《高级机器学习》课，2026-09-17 当堂布置。**

### 作业完整内容（唐老师原话）

> "把大模型全链路亲手走一遍："

- 🔹 **从零写 Tokenizer + Transformer，端到端训一个 0.1B**
- 🔹 **手写 Triton attention kernel，自己测多卡训练和推理增益**
- 🔹 **从 raw dump 洗语料，拟合 scaling law 再外推**
- 🔹 **同一基座上把 SFT、DPO、RLVR 做对照**
- 🔹 **最后搭可验证环境 + harness，训长程 Agent，还鼓励 self-judge loop**

### 课程节奏

| 周 | 交付 |
|---|---|
| W1–W3 | HW1 Foundations |
| W4–W5 | HW2 Systems |
| W6–W7 | HW3 Data + Scaling |
| W8 | Final Project Proposal |
| W9–W11 | HW4 Post-Training |
| W10 | Final Mid Report |
| W12–W15 | FINAL Agent / Harness / Verifier |
| W16 | **现场 demo** + NeurIPS 格式论文 |

### 评分

- **40%** 作业（HW1–HW4）
- **60%** 大项目（FINAL）
- **2–3 人组队**
- **英文，NeurIPS 格式**
- W16 现场 demo（live run）

### 为什么"仓库本身就是作品"

唐老师给这门课的核心命题是 *"A Ladder That Climbs the Course"* —— **前一个作业产出的东西，成为后一个作业的输入**。做完 16 周后，仓库本身就是作品，不是五个互不相关的课程作业。

这个原则在本仓库的具体落点：

- **HW1 的 Transformer** 进 `src/model/`，**HW2 不复制**它，而是在 `src/kernels/` 替换 attention 路径。
- **HW1 的 tokenizer** 进 `src/tokenizer/`，**HW3 不重写**它，而是在同一个 module 上重训。
- **HW3 的 base checkpoint** 被 **HW4** 直接继承。
- **HW4 的 model** 是 **FINAL** agent 的底座。

任何新的研究点（一个新 attention kernel、一种新的 data filter、一个新的 RL 算法、一种新的 self-judge 设计）**不开新 repo**，开一个 `extensions/EXT-NNN-*/`，复用 `src/`，遵循同样的研究模板。

### 边界守住

唐老师明确：**不要一开始就加 HW5/HW6**。HW1–HW3 的目标是**亲手掌握底层**，不是让你在 0.1B 上卷 SOTA。真正值得长期投入增量研究的地方，从 HW4 的 Verifier/RLVR 开始，到 FINAL 的 Environment/Harness/Evaluation/Self-Improvement。前半程把地基走通，后半程把仓库变成自己的研究方向。

### PPT 留档

> 🖼 **TODO（下一次 commit）**：把唐老师 2026-09-17 那堂课的 PPT 放到 `docs/course-origin/tang-2026-09-17-aml-ladder.pdf`，把 *"A Ladder That Climbs the Course"* 那张图单独截出来放 `docs/course-origin/slide-ladder.png`，在首页 hero 区引用。

---

## License

Apache 2.0