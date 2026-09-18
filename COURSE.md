# From Token to Agent

> 从 Tokenizer、Transformer、预训练与 Scaling，一路走到 SFT、Preference Learning、RLVR、Agent RL、Harness、Memory 与 Self-Evolution。
>
> 目标不是完成几个 Demo，而是亲手走一遍现代大模型系统从 **Token 到 Agent** 的完整能力链路。

---

# 一、课程目标

这套课程参考唐杰老师课程的 16 周结构进行实践，并结合 2026 年大模型、Post-Training 与 Agent 方向的发展重新设计实验。

整个课程围绕一条核心主线展开：

> **labels → word structure → next-token → preferences → verifiers → environment → self-judge**

也就是学习信号逐渐发生变化：

```text
人工标签
↓
语言本身的结构
↓
Next-Token Prediction
↓
人类 / AI Preference
↓
程序化 Verifier
↓
Environment Feedback
↓
Self-Judge / Self-Evolution
```

背后的核心趋势是：

> **学习信号越来越便宜，越来越自动化，人逐渐退出数据生产与反馈闭环。**

16 周之后，希望真正具备以下能力：

```text
Raw Data
↓
Tokenizer
↓
Transformer
↓
Pretraining
↓
Scaling
↓
Systems Optimization
↓
Data Engineering
↓
SFT
↓
Preference Learning
↓
RLVR
↓
Agent RL
↓
Harness
↓
Memory
↓
Self-Evaluation
↓
Self-Improvement
```

---

# 二、课程原则

## 2.1 不缩减学习目标，只缩减算力规模

这套课程不要求使用 Frontier Model 规模复现。例如：

- 用约 0.1B 模型学习预训练与 Scaling；
- 用 1.5B～9B 的开源 Base Model 学习 Post-Training；
- 用小型环境复现 Agent RL；
- 用小规模 GPU 实验理解 Distributed Training。

缩小的是：

- **模型规模**
- **数据规模**
- **GPU 数量**
- **训练时间**

不应该省掉的是：

- **Tokenizer**
- **Transformer**
- **Training Loop**
- **Kernel**
- **Scaling**
- **Data**
- **SFT**
- **DPO**
- **RLVR**
- **Environment**
- **Harness**
- **Memory**
- **Self-Judge**
- **Evaluation**

---

# 三、什么需要 From Scratch？

这门课不是要求所有基础设施都从零造。原则是：

> **核心机制亲手实现一次；规模化训练和系统基础设施使用成熟框架。**

| 模块 | 要求 |
|---|---|
| Tokenizer | 自己实现 |
| BPE Training | 自己实现 |
| Transformer | 自己实现 |
| Attention / RoPE / RMSNorm / SwiGLU | 自己实现 |
| 基础 Training Loop | 自己实现 |
| AdamW / LR Schedule / Grad Clip | 至少理解并实现教学版本 |
| Triton Attention Kernel | 自己实现 |
| Distributed Training | 使用 PyTorch DDP / FSDP / Megatron 等 |
| Serving | 使用 vLLM / SGLang |
| Data Pipeline | Pipeline 与规则自己设计，底层库可使用 |
| SFT | 自己理解 Loss 与 Mask；正式训练可使用成熟框架 |
| DPO | 自己实现核心 Loss；规模训练可使用 TRL 等 |
| RLVR | 自己实现教学版核心流程；规模训练使用 slime / verl 等 |
| Agent Harness | 自己实现最小版本 |
| Environment | 自己设计 |
| Verifier | 自己设计 |
| Memory | 自己实现与做 Ablation |
| Self-Judge | 自己实现并校准 |
| Evaluation | 自己定义并实现核心指标 |

判断标准不是：

> "有没有使用框架？"

而是：

> **如果把框架拿掉，我是否知道它替我做了什么？**

---

# 四、课程结构

## 16 周 Topic

| 周 | Topic |
|---|---|
| W1 | Three Paradigm Shifts |
| W2 | Architecture Revisited |
| W3 | Training Dynamics & Scaling Laws |
| W4 | Compute, Kernels, Parallelism |
| W5 | Economics of Inference |
| W6 | Pretraining Data |
| W7 | Synthetic Data & Governance |
| W8 | SFT & Distillation |
| W9 | Preference Learning |
| W10 | RLVR & Reasoning |
| W11 | Agent RL: Long-Horizon |
| W12 | Agent Foundations |
| W13 | Memory & Continual Learning |
| W14 | Self-Evaluation & Evolution |
| W15 | Evaluation & Safety |
| W16 | What Comes After LLMs? |

---

# 五、正式作业

课程共有四个 Homework 和一个 Final Project。

| 作业 | 目标 |
|---|---|
| **HW1 Foundations** | 从零实现 Tokenizer + Transformer，并端到端训练约 0.1B 模型 |
| **HW2 Systems** | 手写 Triton Attention Kernel，测试训练、推理与多卡收益 |
| **HW3 Data + Scaling** | 从 Raw Dump 构建 Corpus，拟合 Scaling Law 并验证外推 |
| **HW4 Post-Training** | 在同一个 Base Model 上比较 SFT / DPO / RLVR，并扩展 Agent RL |
| **Final Project** | Verifiable Environment + Harness + Long-Horizon Agent + Self-Judge |

Final Project 从 W8 开始并行推进：

- **W8** Proposal
- **W10** Mid Report
- **W16** Final Paper + Live Demo

---

# 六、W1 — Three Paradigm Shifts

## 核心问题

- 大模型的学习信号是如何从人工标签，逐步变成数据自身提供的监督？
- 为什么 Tokenization 会成为现代语言模型的第一层基础？

## 学习内容

理解：

- Supervised Learning
- Self-Supervised Learning
- Character / Word / Subword
- BPE
- Next-Token Prediction
- Learning Signal 的演化

重点理解整套课程的 Ladder：

```text
labels
→ word structure
→ next-token
→ preferences
→ verifiers
→ environment
→ self-judge
```

## 实现

从零实现：

- Character Tokenizer
- Word Tokenizer
- BPE Tokenizer

至少自己实现：

- BPE training
- encode
- decode
- vocabulary
- merge rules

**不能直接用 SentencePiece 完成作业。** 可以在完成之后拿 SentencePiece / tiktoken 做 Benchmark。

## 实验

选择混合数据：

- 中文
- 英文
- 代码

比较：

- vocabulary size
- sequence length
- compression ratio
- rare token fragmentation
- 中文 / 英文 / Code 的差异

## 交付

```
src/token_to_agent/from_scratch/tokenizer/
weeks/w01-paradigm-shifts/
```

输出：

- tokenizer 实现
- 单元测试
- 对比实验
- signal-ladder.md

## 验收标准

必须证明：

- **encode → decode** 能够正确 round-trip。

并给出定量 Tokenization 实验。

---

# 七、W2 — Architecture Revisited

## 核心问题

- 今天的大模型和 2017 年 Transformer 到底有什么变化？
- 哪些改动改变了模型训练和推理效率？

## 学习内容

重点：

- Decoder-only Transformer
- Pre-Norm
- RMSNorm
- RoPE
- MHA
- MQA
- GQA
- SwiGLU
- KV Cache
- MoE

同时理解：

- **Total Parameters**
- **Activated Parameters**
- **FLOPs / Token**

之间的区别。

## 实现

从零实现：

- Embedding
- RMSNorm
- RoPE
- Causal Attention
- GQA / MHA
- SwiGLU
- Transformer Block
- LM Head
- Generation

形成一个完整的 Decoder-only Transformer。

## 实验

至少完成两个 Ablation：

- **LayerNorm vs RMSNorm**
- **MHA vs GQA**

比较：

- Parameter
- FLOPs
- Memory
- Tokens/s
- Validation Loss

## 延伸

可增加：

- Tiny MoE
- Sparse Attention
- Multi-Token Prediction

## 智谱研究连接

阅读 **GLM-4.5 / GLM-5** 的模型结构。重点不是复现数百 B 参数，而是理解：

- MoE
- Hybrid Reasoning
- Sparse Attention
- Activated Parameters
- Inference Cost

这些设计为什么出现。

---

# 八、W3 — Training Dynamics & Scaling Laws

## 核心问题

- 为什么 Transformer 能稳定训练？
- 模型规模、数据规模、训练计算量之间存在什么规律？

## 实现

完成完整 Pretraining Pipeline：

```text
Raw Text
↓
Tokenizer
↓
Dataset Shard
↓
DataLoader
↓
Transformer
↓
Forward
↓
Loss
↓
Backward
↓
Optimizer
↓
Checkpoint
↓
Evaluation
```

训练一个约 **100M Parameters** 的模型。

### Training Loop

这一阶段基础训练循环必须自己实现，包括：

- forward
- loss
- backward
- gradient accumulation
- gradient clipping
- AdamW
- LR warmup
- LR decay
- checkpoint
- evaluation

## 实验

建议训练三个模型：

- ~20M
- ~50M
- ~100M

记录：

- Train Loss
- Validation Loss
- Gradient Norm
- Learning Rate
- Tokens Seen
- Tokens/s
- GPU Memory
- GPU Hours

做第一次 Scaling Pilot。

## HW1 — Foundations

提交：

- Tokenizer
- Transformer
- Training Loop
- ~0.1B Pretraining
- Architecture Ablation
- Training Dynamics

必须支持：

```bash
python scripts/train.py --config configs/pretrain/100m.yaml
```

进行可复现实验。

---

# 九、W4 — Compute, Kernels, Parallelism

## 核心问题

- Transformer 的时间到底花在哪里？
- 模型训练性能问题有多少来自算法，又有多少来自 GPU 系统？

## 学习内容

理解：

- GPU Memory Hierarchy
- HBM
- SRAM
- Arithmetic Intensity
- Kernel Launch
- Attention Complexity
- FlashAttention
- Triton
- DDP
- FSDP
- Tensor Parallel

## 实现

亲手实现一个 **Triton Attention Kernel**。

比较：

- Naive PyTorch Attention
- PyTorch SDPA
- Triton Attention

## 实验

不同 Sequence Length：

- 512
- 2048
- 8192

测试：

- Forward latency
- Backward latency
- Peak memory
- Numerical error
- Throughput

至少跑一次真实 Multi-GPU 实验。可以使用 PyTorch DDP / FSDP / Megatron。**这里不要求自己重新实现 Distributed Training。**

---

# 十、W5 — Economics of Inference

## 核心问题

- 模型跑快了，真的更便宜了吗？
- 为什么训练 FLOPs 不是推理成本的全部？

## 学习内容

理解：

- Prefill
- Decode
- TTFT
- TPOT
- Continuous Batching
- KV Cache
- Prefix Cache
- Speculative Decoding
- Serving Scheduler

## 实现

选择 **vLLM** 或 **SGLang** 搭建 Serving。自己实现 Load Generator。

## 实验

改变：

- Concurrency
- Batch Size
- Prompt Length
- Generation Length

测：

- TTFT
- TPOT
- P50
- P95
- Tokens/s
- GPU Memory
- Request Throughput

进一步计算：

- Cost / 1M Input Tokens
- Cost / 1M Output Tokens
- Cost / Successful Task

## HW2 — Systems

HW2 最终需要同时回答：

> **How Fast?** + **How Expensive?**

提交：

- Triton Kernel
- Multi-GPU Benchmark
- Serving Benchmark
- Inference Economics

---

# 十一、W6 — Pretraining Data

## 核心问题

Raw Web / Document Data 如何变成值得花 GPU 训练的 Corpus？

## 学习内容

理解：

- Parsing
- Normalization
- Language Identification
- Quality Filtering
- Exact Dedup
- Near Dedup
- Contamination
- Data Mixture
- Data Provenance

## 实现

完成：

```text
Raw Dump
↓
Parse
↓
Normalize
↓
Language Filter
↓
Quality Filter
↓
Exact Dedup
↓
Near Dedup
↓
Contamination Check
↓
Mixture
↓
Tokenizer
↓
Training Shards
```

**不能直接把一个已经清洗好的 HuggingFace Dataset 当作 HW3 的全部数据工作。**

## 实验

维护完整 Data Ledger：

| Stage | Docs | Tokens | Retention | Duplicate Rate |
|---|---|---|---|---|
| Raw        | ✓ | ✓ | 100% | — |
| Parsed    |   |   |    |   |
| Filtered  |   |   |    |   |
| Deduplicated |   |   |    |   |
| Final     |   |   |    |   |

训练小模型比较：

- Raw-ish Data
- Clean Data

观察 Data Quality 如何影响模型。

---

# 十二、W7 — Synthetic Data & Governance

## 核心问题

- Synthetic Data 什么时候能增强模型？
- 什么时候会放大错误、偏差和数据污染？

## 学习内容

理解：

- Synthetic Task Generation
- Rejection Sampling
- LLM Judge
- Verifier Filtering
- Data Provenance
- Contamination
- Model Collapse
- Governance

## 实现

搭建最小 Synthetic Data Factory：

```text
Task Generator
↓
Candidate Task
↓
Solver
↓
Verifier / Judge
↓
Filter
↓
Training Data
```

## 实验

比较：

- Real Only
- Clean Real
- Clean Real + Synthetic

观察 downstream performance。

## Scaling Law 正式实验

这里把 W3 的 Scaling Pilot 升级。

选择若干训练点：

- 20M
- 50M
- 100M

或多个 Token Budget。然后：

```text
Fit
↓
Predict Unseen Point
↓
Actually Train
↓
Prediction Error
```

必须真正测试外推准确度。

## HW3 — Data + Scaling

最终提交：

- Raw Data Pipeline
- Data Card
- Synthetic Data Experiment
- Scaling Law
- Held-out Prediction

---

# 十三、W8 — SFT & Distillation

## 核心问题

- Base Model 如何第一次真正学习"完成任务"？
- 如何把强模型能力迁移给较弱模型？

## 模型策略

从这里开始进入第二条模型线：

- **Scratch Track**（~100M）继续用于 Architecture / Training / Scaling / Systems
- **Post-Training Track**：选择 **1.5B ~ 9B** 规模的开源 Base Model，用于 SFT / DPO / RLVR / Agent RL

原因是 100M 模型可能根本缺乏完成复杂任务的基础能力。

## 学习内容

理解：

- Instruction Tuning
- Chat Template
- Loss Mask
- Response Mask
- Distillation
- Reasoning Data
- Trajectory Data

## 实现

先自己实现最小：

- masked causal LM loss

理解 SFT 到底改变了哪些 Token 的 Loss。正式训练可以使用成熟框架。

## 实验

比较：

- Base
- SFT
- Distilled Model

统一 Eval。

## Final Project Proposal

W8 提交 Proposal。必须定义：

- Problem
- Motivation
- Environment
- Verifier
- Harness
- Base Model
- Evaluation
- Compute Budget
- Hypothesis

---

# 十四、W9 — Preference Learning

## 核心问题

- 没有唯一正确答案时，模型该从什么 Signal 学习？
- Preference 真的是可靠的 Ground Truth 吗？

## 学习内容

理解：

- Preference Pair
- Reward Model
- Bradley-Terry
- DPO
- Judge Bias
- Calibration

## From Scratch

自己实现核心 **DPO Loss**。理解：

- chosen log probability
- rejected log probability
- reference policy
- policy ratio
- DPO objective

## 正式训练

可以使用 TRL / LLaMA-Factory 等成熟 trainer 进行规模训练。

## 实验

同一个 Base Model：

- Base
- SFT
- DPO

统一 Prompt Distribution 与 Eval。

同时比较：

- Human Rule
- Verifier
- LLM Judge
- Reward Model

之间的一致性。至少找到一些 **Preference ≠ Correctness** 的案例。

---

# 十五、W10 — RLVR & Reasoning

## 核心问题

- 当 Reward 从 Preference 变成可验证结果后，会发生什么？
- 为什么 RLVR 能学到 SFT 没学到的行为？

## 学习内容

理解：

- RLVR
- GRPO
- Group Sampling
- Reward Variance
- Difficulty Sampling
- Outcome Reward
- Process Reward
- Reward Hacking

## From Scratch

实现一个小型教学版 RLVR：

```text
Prompt
↓
K Rollouts
↓
Verifier
↓
Reward
↓
Group Advantage
↓
Policy Loss
↓
Update
```

规模化训练不要求从零实现 RL Infra。正式实验可以使用 slime / verl / TRL 等。

## Domain

至少选两个 Verifiable Domain：

- **Math**（Verifier → Exact Answer）
- **Code**（Verifier → Unit Test）

## 实验

严格比较：

- Same Base
- Same Prompt Distribution
- Same Eval
- SFT vs DPO vs RLVR

记录：

- Pass@1
- Reward
- Output Length
- Training Tokens
- GPU Hours
- Reward Variance
- Difficulty Bucket

## Process Reward Extension

可以进一步比较：

- Outcome Reward
- Process Reward
- Outcome + Process

推荐阅读：**ReST-MCTS* / TDRM**。

## Final Mid Report

W10 提交 Final Project Mid Report。

---

# 十六、W11 — Agent RL: Long-Horizon

## 核心问题

RL 从"生成一个答案"变成"在环境中执行很多步"以后，问题发生了什么变化？

```text
普通 RLVR          Agent RL
─────────          ─────────
Prompt             Task
↓                  ↓
Response           Action
↓                  ↓
Verifier           Environment
                   ↓
                   Observation
                   ↓
                   Action
                   ↓
                   ...
                   ↓
                   Terminal State
                   ↓
                   Reward
```

## 学习内容

理解：

- Trajectory
- Long-Horizon Credit Assignment
- Sparse Reward
- Environment Interaction
- Tool Use
- Multi-Turn Rollout
- Agentic RL
- Async RL

## 实现

把 Final Project Environment 真正接进 Rollout。至少跑一个小型 Agent RL Experiment。

> 如果完整规模计算量过大：**缩小模型和环境，而不是省掉 RL Update。**

## 实验

改变：

- Max Turns
- Token Budget
- Tool Budget
- Time Budget

观察：

- Success Rate
- Interaction Turns

同时分析：

> 更多 Interaction 到底是在增加能力，还是只产生更长的失败轨迹？

## HW4 — Post-Training

HW4 汇总：**SFT + DPO + RLVR + Agentic Extension**，要求同一套 Base / Task Distribution / Eval / Cost Accounting。

---

# 十七、W12 — Agent Foundations

## 核心问题

- LLM 和 Agent 之间到底多了什么？
- 同一个模型为什么换一个 Harness 后效果可能变化巨大？

## 学习内容

理解：

- Agent Loop
- Tool Schema
- Action Parser
- Context Manager
- Retry
- Budget
- Sandbox
- Termination
- Trace

## 实现

这一周必须自己实现一个最小 Harness。**不能直接 `LangGraph(...)` 然后认为完成了 Agent Foundations。**

最低结构：

```text
Tools
↑
│
User → Harness → Model
│
├─ Parser
├─ Context
├─ Retry
├─ Budget
├─ Sandbox
├─ Termination
└─ Trace Logger
```

## 实验

这是这一周最重要的实验：

> **Same Model / Same Tasks / Same Tools，比较 Harness A vs Harness B vs Harness C。**

比较：

- Task Success
- Tool Error
- Steps
- Tokens
- Latency
- Cost

必须证明：**Harness 本身也是能力的一部分。**

推荐参考：**AgentTuning / AgentBench**。

---

# 十八、W13 — Memory & Continual Learning

## 核心问题

- Context 和 Memory 有什么区别？
- Agent 如何积累经验，同时避免历史经验污染当前任务？

## Memory 分类

至少区分：

- **Working Memory** — 当前任务上下文
- **Episodic Memory** — 过去的完整 Episode / Trajectory
- **Procedural Memory** — 从过去经验中总结出的 Skill / Rule / Procedure / Strategy

## 实现

至少三种条件：

- No Memory
- Raw Trajectory Retrieval
- Summary / Procedural Memory

## 实验

评价：

- New Task Success
- Old Task Regression
- Retrieval Precision
- Context Cost
- Negative Transfer
- Forgetting

不能把"接入 Vector DB"直接等同于"完成 Memory"。**Memory 必须有 Ablation。**

---

# 十九、W14 — Self-Evaluation & Evolution

## 核心问题

- Agent 能否判断自己什么时候做错了？
- 判断错误以后，能否真正改善行为？

## 实现

完整链路：

```text
Agent Trajectory
↓
Self Judge
↓
Failure Diagnosis
↓
Retry / Revision
↓
External Verifier
```

## 实验

比较：

- First Attempt
- Self-Judge Retry
- External-Judge Retry
- Verifier-Guided Retry

## Judge Calibration

必须测：

- Precision
- Recall
- False Positive
- False Negative
- **Judge Score ↔ True Reward Correlation**

不能只展示几个 "Agent Reflection 后答对了" 的案例。

## Evolution

进一步尝试：

```text
Successful Trajectories
↓
Selection
↓
Distillation
↓
New Policy
```

然后测试：Judge Loop 被拿掉之后，模型本身是否真的变强。

推荐连接：**ReST-MCTS* / TDRM / Iterative Self-Distillation**。

---

# 二十、W15 — Evaluation & Safety

## 核心问题

- 一个 Agent 到底怎样才算"更好"？
- 能力、可靠性、成本与安全是否可能互相冲突？

## Evaluation Matrix

至少评价：

| 维度 | 指标 |
|---|---|
| Capability | Task Success |
| Reasoning | Pass@1 / Pass@k |
| Tool Use | Tool Call Correctness |
| Long Horizon | Steps to Success |
| Reliability | Retry / Recovery Rate |
| Efficiency | Tokens / Task |
| System | Latency |
| Economics | Cost / Successful Task |
| Memory | Transfer / Forgetting |
| Judge | Calibration |
| Safety | Unsafe Action Rate |
| Security | Prompt Injection Success |

## Safety Lab

至少构造三个攻击入口：

- Malicious File
- Malicious Webpage
- Malicious Tool Output

测试 Agent 是否：

- follow untrusted instructions
- leak hidden context
- cross permission boundaries
- perform unsafe actions

这里的安全研究重点已经不只是"模型会不会说危险的话"，而是：

> **Agent 会不会因为恶意 Observation 做出真实 Action。**

---

# 二十一、W16 — What Comes After LLMs?

## 核心问题

未来真正被优化的对象还是 Model 吗？还是一个包含：

- Model
- Harness
- Environment
- Memory
- Verifier
- Evaluator

的完整智能系统？

## Final Ablation

建议最终至少形成：

```text
Base Model
↓
+ SFT
↓
+ RLVR
↓
+ Harness
↓
+ Memory
↓
+ Self-Judge
↓
+ Agent RL
```

不是所有模块都一定带来正收益。**负结果同样需要记录。**

## Final Deliverables

### 1. 英文 NeurIPS 风格论文

结构：

- Problem
- Motivation
- Method
- Experimental Setup
- Results
- Analysis
- Limitations
- Safety
- Future Work

### 2. Live Demo

必须真正端到端运行。

### 3. Reproduction

需要提供：

- Environment
- Configs
- Commands
- Checkpoint
- Dataset Version
- Eval Version

### 4. Experiment Registry

保留所有核心实验。

---

# 二十二、Final Project 默认方向

默认推荐：**Verifiable Coding Agent**

```text
Repository
+
Issue
↓
Harness
↓
Model
↓
Read / Search / Edit / Test / Shell
↓
Sandbox
↓
Patch
↓
Unit Tests
↓
Verifier
↓
Reward
```

它天然包含：

- Long-Horizon Interaction
- Tool Use
- Harness
- Environment
- Sandbox
- Verifier
- Agent RL
- Memory
- Self-Judge
- Evaluation
- Safety

因此非常适合作为整套课程最终汇合点。

- 第二选择：Deep Research Agent
- 第三选择：Computer-Use Agent

> **Final Project 最重要的不是场景炫酷，而是：环境是否可验证。**

---

# 二十三、仓库结构

```
from-token-to-agent/
│
├── README.md
├── COURSE.md
├── ROADMAP.md
│
├── weeks/
│   ├── w01-paradigm-shifts/
│   ├── w02-architecture-revisited/
│   ├── w03-training-dynamics-scaling-laws/
│   ├── w04-compute-kernels-parallelism/
│   ├── w05-economics-of-inference/
│   ├── w06-pretraining-data/
│   ├── w07-synthetic-data-governance/
│   ├── w08-sft-distillation/
│   ├── w09-preference-learning/
│   ├── w10-rlvr-reasoning/
│   ├── w11-agent-rl-long-horizon/
│   ├── w12-agent-foundations/
│   ├── w13-memory-continual-learning/
│   ├── w14-self-evaluation-evolution/
│   ├── w15-evaluation-safety/
│   └── w16-what-comes-after-llms/
│
├── assignments/
│   ├── hw1-foundations/
│   ├── hw2-systems/
│   ├── hw3-data-scaling/
│   ├── hw4-post-training/
│   └── final-project/
│
├── src/token_to_agent/
│   │
│   ├── from_scratch/
│   │   ├── tokenizer/
│   │   ├── model/
│   │   ├── training/
│   │   ├── kernels/
│   │   ├── dpo/
│   │   └── rlvr/
│   │
│   ├── systems/
│   │   ├── distributed/
│   │   ├── serving/
│   │   └── profiling/
│   │
│   ├── data/
│   │
│   ├── post_training/
│   │   ├── sft/
│   │   ├── preference/
│   │   └── rl/
│   │
│   ├── agent/
│   │   ├── harness/
│   │   ├── environments/
│   │   ├── tools/
│   │   ├── memory/
│   │   └── self_eval/
│   │
│   └── evaluation/
│
├── tests/
│
├── configs/
│   ├── pretrain/
│   ├── systems/
│   ├── data/
│   ├── sft/
│   ├── dpo/
│   ├── rlvr/
│   └── agent/
│
├── experiments/
│
├── extensions/
│
├── scripts/
│
├── docs/
│   ├── course-origin/
│   ├── architecture/
│   └── reading/
│
└── artifacts/
    ├── datasets/
    ├── checkpoints/
    ├── logs/
    ├── traces/
    └── profiles/
```

---

# 二十四、每周提交规范

每周至少回答四个问题：

1. **What did I learn?** — 这一周理解了什么机制？
2. **What did I build?** — 真正实现了什么？
3. **What did I measure?** — 跑了什么实验？
4. **What did I conclude?** — 结果说明什么？

每一个重要实验必须记录：

- experiment_id
- git commit
- model
- config
- dataset version
- eval version
- seed
- GPU
- GPU count
- training tokens
- generation tokens
- wall clock
- GPU hours
- result

**失败实验不删除。**

---

# 二十五、Extension 规则

课程之外的新论文与新想法统一进入 `extensions/`。

例如：

| Extension | Topic |
|---|---|
| EXT-W02-001 | Tiny MoE |
| EXT-W04-001 | Fused RMSNorm |
| EXT-W05-001 | Prefix Cache Economics |
| EXT-W07-001 | Verifier-Filtered Synthetic Data |
| EXT-W10-001 | Process Reward |
| EXT-W11-001 | Async Agent RL |
| EXT-W12-001 | Harness Ablation |
| EXT-W13-001 | Procedural Memory |
| EXT-W14-001 | Self-Judge Calibration |

每一个 Extension 必须写清：

- **Hypothesis**
- **Baseline**
- **Intervention**
- **Metric**
- **Result**
- **Conclusion**

不能把"我把某篇论文代码跑起来了"作为 Extension 的最终成果。

---

# 二十六、重点研究参考

课程后半段重点关注智谱 / 清华及相关工作：

**GLM-4.5** — 重点：MoE / Reasoning / Agent / Coding / RL / Agentic RL / Self-Distillation / Interaction Scaling

**GLM-5** — 重点：Sparse Architecture / Inference Efficiency / Asynchronous RL / Long-Horizon Agent RL / Agentic Engineering

**slime** — 重点理解 Rollout / Environment / Verifier / Reward / Training Buffer / Policy Update 如何组成一个真正的 Agent RL System

**ReST-MCTS*** — 重点：Search / Process Reward / Self-Training

**TDRM** — 重点：Temporal Consistency / Process Reward / RLVR

**AgentTuning / AgentBench** — 重点：Tool Use / Planning / Interaction / Agent Evaluation

**SCALE-CUA** — 重点：Environment Generation / Verifiable Tasks / Computer Use / Online RL

**Continual Learning** — 重点：Memory / Stability / Plasticity / Forgetting / Continual Adaptation

---

# 二十七、课程最终验收

完成这套课程时，应当能够勾掉：

- [ ] 自己实现 BPE Tokenizer
- [ ] 自己实现 Decoder-only Transformer
- [ ] 自己实现基础 Training Loop
- [ ] 端到端训练约 0.1B 模型
- [ ] 做过 Architecture Ablation
- [ ] 做过 Training Dynamics 分析
- [ ] 做过 Scaling 实验
- [ ] 自己写过 Triton Attention Kernel
- [ ] 做过 Multi-GPU Training
- [ ] 做过真实 Serving Benchmark
- [ ] 计算过 Inference Economics
- [ ] 从 Raw Dump 构建过 Corpus
- [ ] 做过 Dedup / Quality / Contamination
- [ ] 真正做过 Scaling Extrapolation
- [ ] 做过 Synthetic Data Pipeline
- [ ] 做过 SFT
- [ ] 自己实现过 DPO 核心 Loss
- [ ] 真正训练过 DPO
- [ ] 自己实现过教学版 RLVR
- [ ] 真正完成过一次 RLVR Policy Update
- [ ] 比较过 SFT / DPO / RLVR
- [ ] 搭建过 Verifiable Environment
- [ ] 自己实现过最小 Harness
- [ ] 做过 Same Model / Different Harness 对比
- [ ] 跑过 Long-Horizon Agent
- [ ] 做过 Agent RL 或小规模等价实验
- [ ] 实现过 Memory
- [ ] 做过 Memory Ablation
- [ ] 实现过 Self-Judge
- [ ] 对 Self-Judge 做过 Calibration
- [ ] 测过 Agent Cost
- [ ] 测过 Agent Reliability
- [ ] 做过 Prompt Injection / Tool Safety 测试
- [ ] 完成 Final Research Paper
- [ ] 完成 Live Demo

---

# 二十八、最终标准

如果只完成 Tokenizer / Transformer / Pretraining / Scaling，那么这是一个：

> **LLM From Scratch 项目。**

如果只完成 Harness / Tools / Memory / Agent，那么这是一个：

> **Agent Engineering 项目。**

**From Token to Agent 的目标是把两边真正连起来。**

最终需要能够回答：

> 一个 AI 系统的能力究竟是怎样获得的？

从：

- Token Structure
- Next-Token Learning
- Preference
- Verifier
- Environment
- Memory
- Self-Judge

并且不仅能够解释这些概念，还能够：

> **亲手实现、运行实验、测量结果、发现失败，并继续改进它。**