# Extensions

Extensions are **incremental research work on top of an HW**, not new assignments. Each one lives in its own folder and uses the same Problem → Motivation → Method → Results → Reproduction template.

## Numbering

```
EXT-<stage><3-digit-index>

stage = 1 (HW1) | 2 (HW2) | 3 (HW3) | 4 (HW4) | 5 (FINAL)
index = 01, 02, 03, ...
```

Pick the next free number under the stage you belong to.

## Planned (will fill as the course progresses)

### HW1 extensions
- `EXT-101` RoPE
- `EXT-102` RMSNorm
- `EXT-103` SwiGLU
- `EXT-104` Grouped Query Attention (GQA)
- `EXT-105` MuP init
- `EXT-106` Different tokenizer (BBPE, Unigram)

### HW2 extensions
- `EXT-201` Triton Attention forward
- `EXT-202` Triton Attention backward
- `EXT-203` Fused RMSNorm kernel
- `EXT-204` Fused MLP (fused gate+up+down)
- `EXT-205` KV Cache for inference
- `EXT-206` torch.compile baseline
- `EXT-207` Tensor Parallel (single-node)
- `EXT-208` FSDP / ZeRO-2
- `EXT-209` INT8 / FP8 quantization

### HW3 extensions
- `EXT-301` MinHash-based dedup
- `EXT-302` Quality classifier (KenLM perplexity / fastText)
- `EXT-303` Language ID filtering (fastText langid)
- `EXT-304` Data mixture ablation (compute-optimal)
- `EXT-305` Curriculum learning
- `EXT-306` Contamination detection

### HW4 extensions
- `EXT-401` Online DPO
- `EXT-402` GRPO
- `EXT-403` Reward hacking analysis
- `EXT-404` Process reward model (PRM)
- `EXT-405` Best-of-N at inference
- `EXT-406` Judge LLM

### FINAL extensions
- `EXT-501` Memory (episodic / semantic)
- `EXT-502` Self-judge
- `EXT-503` Failure attribution from trajectories
- `EXT-504` On-policy trajectory distillation
- `EXT-505` Multi-agent orchestration
- `EXT-506` RL on agent (Agent RL)

## Folder layout

```
extensions/
└── EXT-XXX-name/
    ├── README.md         ← Problem / Motivation / Method / Results / Reproduction
    ├── src/              ← code (imports from /src/, never duplicates)
    ├── configs/
    ├── experiments/
    └── results/
```

Every extension must:
1. Reuse `src/` — no copy-paste of prior code.
2. State in README what it inherits from its parent HW.
3. Run a comparison vs the parent HW baseline (table or chart).
4. Cite what it's reproducing or extending (paper / blog post / repo).