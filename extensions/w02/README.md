# W2 Architecture Extensions

Planned:

- `EXT-W2-01` Tiny MoE — Total / activated / FLOPs-per-token decomposition
- `EXT-W2-02` Sparse attention prototype (e.g., sliding-window or block-sparse)
- `EXT-W2-03` Multi-token prediction head

See `weeks/w02-architecture-revisited/notes.md` for the architectural motivation.

Each extension lives in its own folder:

```
extensions/w2/ext-w2-NNN-<name>/
    README.md
    src/
    configs/
    experiments/
    results/
```