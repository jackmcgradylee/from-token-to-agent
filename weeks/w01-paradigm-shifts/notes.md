# W1 — notes.md

## The signal ladder

The single most important sentence of the course:

> labels → word structure → next-token → preferences → verifiers → environment → self-judge

Each rung shifts the cost of producing the learning signal — and who (or what) produces it.

| Rung | Signal producer | Cost to produce | Reliability | Failure modes |
|---|---|---|---|---|
| labels | humans | very high | subjective, sometimes wrong | label noise, distribution drift |
| word structure | self-supervised (BPE/word/char statistics) | low (one pass over text) | 100% (lossless) | tokenizer fragmentation on rare words |
| next-token | self-supervised (raw text) | very low | objective | distribution bias, undertraining |
| preferences | humans / LLM judge | medium | inconsistent, biased | judge ≠ ground truth |
| verifiers | programs (unit tests, exact-answer checks) | medium | near-deterministic | only works where reward is programmable |
| environment | environment itself (state transitions, terminal reward) | medium | task-dependent | sparse rewards, credit assignment |
| self-judge | the model itself | low (one extra inference) | calibrated against external verifier or it isn't useful | reward hacking, sycophancy |

## Reading

- See `docs/reading/` (to be populated).
- For the W1 reading list, see `COURSE.md §2 W1`.

## Open questions

- Is "rare-token fragmentation" actually a useful metric for tokenizer comparison, or is "fertility" (tokens / word) better?
- Should the W1 mixed sample be Chinese / English / code, or include math notation too?
- For self-supervised BPE, what's the right minimum corpus size — is 1 MB representative of TinyStories / Wikipedia behavior?