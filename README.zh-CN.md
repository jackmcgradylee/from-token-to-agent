# From Token to Agent · 从分词到智能体

> 从分词和预训练，到 RLVR、长程智能体、自改进——
> 从零搭建现代 AI 栈。

---

## Learning Signal Ladder

```
labels
→ word structure
→ next-token
→ preferences
→ verifiers
→ environment
→ self-judge
```

这是唐杰老师 2026 年秋季清华大学《高级机器学习》课（2026-09-17 当堂布置）所蕴含的能力梯子。每往上一级，"signal 越来越便宜来生产 — 人类再退出一圈"。

下面是那天课的 PPT 6 张合图：

![唐杰 2026-09-17《高级机器学习》课 PPT 6 张合图，含 16 周课程表和 ladder footer](./docs/course-origin/tang-2026-09-17-aml-ppt-collage.jpg)

合图按顺序：LLM 用例金字塔、Scaling ↔ 泛化时间线、开课引入（"让机器像人一样思考"）、2026 研究方向、**16 周课程表**、历史脉络（三次范式转移，1980s → 2026）。

## 进度

```
W01  Three Paradigm Shifts                   Done (smoke)
W02  Architecture Revisited                  In Progress
W03  Training Dynamics & Scaling Laws        In Progress
W04  Compute, Kernels, Parallelism           In Progress
W05  Economics of Inference                  Not Started
W06  Pretraining Data                        Not Started
W07  Synthetic Data & Governance             Not Started
W08  SFT & Distillation                      Not Started
W09  Preference Learning                     Not Started
W10  RLVR & Reasoning                        Not Started
W11  Agent RL: Long-Horizon                  Not Started
W12  Agent Foundations                       Not Started
W13  Memory & Continual Learning             Not Started
W14  Self-Evaluation & Evolution             Not Started
W15  Evaluation & Safety                     Not Started
W16  What Comes After LLMs?                  Not Started
```

```
HW1 — Foundations          In Progress
HW2 — Systems              In Progress
HW3 — Data + Scaling       Not Started
HW4 — Post-Training        Not Started
Final Project              Not Started
```

详细当前状态和下一步具体动作见 [`ROADMAP.md`](./ROADMAP.md)。

## 课程

完整 16 周排程、每周交付物、Pass Criteria、Extension 规范都在 [`COURSE.md`](./COURSE.md)。这份文件是 syllabus 的 source of truth，按"基本冻结"对待。

## 仓库结构

```
weeks/        学习过程 — W1 → W16，每周一个目录
assignments/  阶段提交 — HW1、HW2、HW3、HW4、Final
src/          唯一正式代码区，按 src/token_to_agent/<module>/ 组织
tests/        单元 + 正确性测试，与 src/ 同构
configs/      可复现实验配置
experiments/  每次真实运行的证据（config / metrics / metadata / README）
extensions/   课程要求之外的增量研究，按 EXT-W<N>-<idx> 编号
docs/         系统图、论文阅读笔记、课程缘起
artifacts/    本地产物（checkpoints / datasets / logs / traces / profiles）
scripts/      CLI 入口（train / evaluate / benchmark / generate / serve）
```

纪律：**代码积累，绝不复制**。一周写进 `src/token_to_agent/<module>/`；HW 的 `report.md` 只引用代码和实验证据，不复制任何一份。

## 快速开始

```bash
bash scripts/setup_env.sh

# 训练 toy 0.1B scratch baseline（smoke test，Jetson CPU 上约 5 分钟）
PYTHONPATH=. .venv/bin/python scripts/train.py --config configs/pretrain/toy-5m.yaml

# 生成样例
PYTHONPATH=. .venv/bin/python scripts/generate.py \
    --ckpt artifacts/checkpoints/hw1-toy-5m/model.pt \
    --tokenizer artifacts/checkpoints/hw1-toy-5m/tokenizer.json \
    --prompts-file experiments/w03/exp-001-toy-baseline/prompts.txt \
    --out experiments/w03/exp-001-toy-baseline/samples.txt

# 跑所有测试
PYTHONPATH=. .venv/bin/python tests/tokenizer/test_bpe.py
PYTHONPATH=. .venv/bin/python tests/model/test_transformer.py
PYTHONPATH=. .venv/bin/python tests/kernels/test_attention.py
```

## 边界

- **不是 16 个孤立的周项目**。代码积累；周是学习单位，不是交付单位。
- **不是在 0.1B 上卷 SOTA**。前 7 周是**理解**每一级，不是打榜。真正值得长期投入研究的地方，从 W8 的 SFT/DPO/RLVR 开始，到 W16 的环境/harness/自评判/self-improvement。
- **不是第一天就搭"什么都会的通用 Agent 框架"**。FINAL 第一版 = 一个可验证环境 + 一个 harness + 一个 agent。泛化走 Extension 路线。
- **不下载预训练权重 / 不下载大数据集**。本仓库 implement-first、framework-second——机制在这里写，不去 fetch。

## License

Apache 2.0 — 见 [`LICENSE`](./LICENSE)。

## 致谢

本仓库是 **唐杰老师（THUDM / Z.ai）2026-09-17 在清华大学《高级机器学习》课正式布置** 的 16 周作业的作业仓库。7 级 ladder、Problem → Motivation → Method → Results 研究模板、"代码积累，绝不复制"纪律、"不加 HW5/HW6 — 走 Extension 路线"规则——都直接来自那次布置。

`COURSE.md §7` 阅读列表引用 THUDM / Z.ai 一线工作（GLM-4.5 / GLM-5 / slime / ReST-MCTS* / TDRM / AgentTuning / AgentBench / ScaleCUA / INFTY）作为方向参考，不是复刻清单。