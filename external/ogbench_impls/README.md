# OGBench `impls` additions

The latent-space trainers in `offline_to_online/` (`train_hiql_acfql_latent.py`,
`train_wgsp_lsg.py`, `train_wgsp_lsg_phase2.py` and their eval scripts) import the
agents and utilities of the OGBench reference implementations by inserting
`~/ogbench/impls` at the front of `sys.path`. Those agents live in the ogbench
clone, not in this repository, so they are vendored here for reproducibility.

Contents:

- `agents/hiql_acfql.py` — HIQL with an ACFQL flow-matching chunked low level
  (new agent, written for this project on top of ogbench's `hiql.py`).
- `agents/wgsp_lsg.py` — the Latent Subgoal Planning (LSP / WGSP-LSG) agent:
  flow HL + flow LL + IQL value, with the Phase-2 world-model-grounded update
  modes (`hl_mode = awr | hrf | real | frozen`).
- `ogbench_impls_changes.patch` — the diff against the upstream ogbench repo
  that the agents depend on: `ActorVectorField`/`FourierFeatures` flow networks,
  config-gated action chunking in `HGCDataset`, a chunked hierarchical eval
  loop, an `Identity` encoder, and agent registration.

## Setup

```bash
git clone https://github.com/seohongpark/ogbench ~/ogbench
cd ~/ogbench
git checkout 1d4140997f60c52c6fb0702ec100dc988b18c548   # commit the patch was made against
git apply /path/to/this/repo/external/ogbench_impls/ogbench_impls_changes.patch
cp /path/to/this/repo/external/ogbench_impls/agents/*.py impls/agents/
```

The trainers expect the clone at `~/ogbench` (see `_OGB_IMPLS` at the top of
`offline_to_online/train_wgsp_lsg.py` and `train_hiql_acfql_latent.py`).
