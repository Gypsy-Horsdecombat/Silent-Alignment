## GPT Baseline Results (v2.2)

This repository includes a completed GPT-only baseline run of the
Silent Alignment v2.2 protocol.

- Model: GPT (seeded, temperature = 0.7)
- Tasks: 83 (committee-curated, veto-based inclusion)
- Embeddings: text-embedding-3-large
- ε = 0.80, τ = 0.60
- Observed silent overlap rate: 96.4%
- Permutation null p-value: 0.0033 (K = 300)

Results are provided in `results/gpt_v2_2_run_N83/` and serve as the
reference baseline for cross-provider replication (Claude, Grok).
