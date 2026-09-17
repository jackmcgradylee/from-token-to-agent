# Extensions

Extensions are **incremental research work on top of a week**, not new assignments. Each one lives in its own folder and answers five questions: Hypothesis / Baseline / Intervention / Metric / Conclusion.

## Numbering

```
EXT-W<N>-<idx>

<N>  = which week it branches off (1–16)
<idx> = 3-digit zero-padded counter

Examples:
  EXT-W2-01   HW1-extension branching from W2 (Architecture)
  EXT-W4-01   HW2-extension branching from W4 (Kernels)
  EXT-W10-01  HW4-extension branching from W10 (RLVR)
```

Pick the next free number under the week you belong to.

## Folder layout

```
extensions/
└── w<N>/
    └── ext-w<N>-<idx>-<name>/
        ├── README.md         ← Hypothesis / Baseline / Intervention / Metric / Conclusion
        ├── src/              ← optional; imports from /src/token_to_agent/, never duplicates
        ├── configs/
        ├── experiments/
        └── results/
```

## Discipline

Every extension must:
1. Reuse `src/token_to_agent/` — no copy-paste of prior code.
2. State in README what it inherits from its parent week.
3. Run a comparison vs the parent week baseline (table or chart).
4. Cite what it's reproducing or extending (paper / blog / repo) when applicable.

## Planned (will fill as the course progresses)

See each week's `notes.md` for the planned extension list. The plan is also re-stated in `COURSE.md §5`.

When a new paper appears, the workflow is:

> "This looks like it belongs to W10 RLVR. Open `extensions/w10/ext-w10-NNN-<name>/`."

**Do not create a new project for every new paper.** Open an extension.