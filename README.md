# Leveraging Latent World Models for Offline Reinforcement Learning

Code for my Imperial College London individual project. The thesis (in
[`thesis/`](thesis/)) asks one question: **how can a learned latent world model be
used to improve goal-conditioned control in high-dimensional robotic
manipulation?** Everything is evaluated on the [OGBench](https://github.com/seohongpark/ogbench)
visual manipulation suite (`cube-single`, plus `scene` and `puzzle-3x3` for
generalisation), with a JEPA-style latent world model (LeWM) providing the
192-D representation and the latent dynamics.

The short version of the findings (Chapter 5 of the thesis):

- **The world model is not a usable replacement environment.** Every variant of
  "run online RL inside the world model" (sparse/dense rewards, latent
  anchoring, joint predictor–policy training, uncertainty penalties) ends below
  the offline policy it started from.
- **It is a good short-horizon planner.** Best-of-N, gradient-refined, and
  FMQ (flow-map Q-guidance) MPC on a *frozen* policy lift a 81.6% offline
  checkpoint to 88–90%, with performance peaking at 1–2 world-model steps and
  degrading beyond that, exactly as the predictor-drift diagnostic predicts.
- **It is most useful as a data generator for the actor only.** Interleaved
  training on MPC-relabelled actions reaches **96.0%** with FMQ inference
  (Q-chunking reference: 92.8%), and online fine-tuning on imagined transitions
  works if and only if the critic is frozen (90.7% final / 94.8% peak vs. a
  30.7% collapse when the critic is also updated).
- **Action conditioning predicts transfer.** The same recipe gives large gains
  on `cube-single` (8.0× action-conditioning ratio), marginal gains on `scene`
  (1.4×), and nothing on `puzzle-3x3` (1.1×).
- **Hierarchy: offline is strong, world-model grounding is not.** The
  flow-policy hierarchical baseline (LSP Phase 1) reaches 88.0% five-task
  success, slightly above end-to-end HIQL (87.0%), but every Phase-2 attempt to
  improve the high level with world-model rollouts destabilises it.

---

## Repository layout

| Path | What it is |
|---|---|
| `jepa.py`, `module.py` | LeWM/JEPA architecture: ViT-Tiny encoder, autoregressive Transformer predictor, action embedder, SIGReg loss modules. |
| `worldmodel/` | World-model training: `wm_train.py` (Lightning + Hydra, configs in `worldmodel/config/`), `finetune_wm_on_play.py` (fine-tune on the offline play data → the `lejepa_play_ft_full` checkpoint used everywhere), `finetune_jepa.py`, `wm_eval.py`, `export_jointtrained_wm.py`. |
| `data_utils/` | Dataset pipeline: `convert_ogbench_npz.py` (OGBench NPZ → HDF5), `refilm_dataset.py` (re-render 64×64 data at 224×224 by replaying qpos/qvel — and `button_states` for scene/puzzle), `reencode_play_dataset.py` (build the 192-D latent cache with a given encoder), plus cache/goal builders. |
| `analysis/` | World-model diagnostics: predictor drift, action conditioning, latent geometry, CEM planning baseline. |
| `offline_to_online/` | The main experimental codebase, **forked from the official Q-chunking repo** (see credits). Contains the ACFQL agent, the JAX port of the WM predictor (`wm_jax.py`), the WM-as-environment pipeline, all MPC planners, the online-in-WM trainers, the static dataset generators, and the LSP (latent subgoal planning) trainers. |
| `wgsp/` | Hierarchical experiments at the representation level: PyTorch HIQL ports (baseline / end-to-end / LeWM-encoder variants), action decoders, WGSP and its FMQ/GRPO distillation variants, hierarchical SAC. |
| `sac/` | Flat and hierarchical SAC baselines trained inside the WM or the real env (appendix retrospective), plus CPU smoke tests. |
| `external/ogbench_impls/` | Vendored copies of the two custom agents (`hiql_acfql.py`, `wgsp_lsg.py`) and the patch that the latent trainers need applied to a local [ogbench](https://github.com/seohongpark/ogbench) clone. |
| `thesis/` | Full LaTeX source of the thesis (`main.tex`), including all chapters, figures, and the scripts that generate the background figures. |
| `bc_policy.py`, `latent_env.py` | Early flat-BC policy and latent-environment wrapper (legacy, kept for the early SAC/HER experiments). |

Code names vs. thesis names: the thesis calls the hierarchical method of
Chapter 4 **Latent Subgoal Planning (LSP)**; in the code it is `wgsp_lsg`
("world-model-grounded subgoal planning, latent subgoals"). `FMQ` is the
flow-map Q-guidance planner of Chapter 3.

---

## Adapted third-party code

This repository builds directly on several public codebases. Credit where it
is due:

- **Q-chunking** — [`offline_to_online/`](offline_to_online/) is a fork of the
  official implementation of *Reinforcement Learning with Action Chunking*
  (Qiyang Li, Zhiyuan Zhou, Sergey Levine, arXiv:2507.07969),
  <https://github.com/ColinQiyangLi/qc>. The ACFQL agent
  (`agents/acfql.py`), the `utils/` and `envs/` scaffolding, and the original
  `main.py` offline-to-online loop are theirs (their README and LICENSE are
  kept in that folder); the WM environment, MPC planners, JAX WM port,
  goal-conditioned trainers, and everything `train_*`/`eval_*` beyond
  `main.py` were added for this project. The published Q-chunking
  `cube-single-task1` result (92.8%) is the reference baseline in the thesis.
- **HIQL / OGBench reference implementations** — the PyTorch HIQL trainers in
  [`wgsp/`](wgsp/) (`train_hiql_baseline.py`, `train_hiql_endtoend.py`,
  `train_hiql_lewm.py`, …) are faithful ports of `impls/agents/hiql.py` from
  <https://github.com/seohongpark/ogbench> (the reference implementation of
  *HIQL: Offline Goal-Conditioned RL with Latent States as Actions*, Park,
  Ghosh, Eysenbach, Levine — official repo <https://github.com/seohongpark/HIQL>),
  with the visual encoder swapped for the frozen LeWM ViT. The JAX agents in
  `external/ogbench_impls/agents/` are likewise written on top of the ogbench
  `impls` agent/utils framework and run inside a patched ogbench clone.
- **FQL** — `offline_to_online/agents/iql_chunk.py` and `agents/rebrac_chunk.py`
  are vendored, near-verbatim, from the IQL and ReBRAC reference agents in
  <https://github.com/seohongpark/fql> (*Flow Q-Learning*, Park, Li, Levine),
  adapted only to treat the 25-D action chunk as a flat action.
- **LeWM (LeWorldModel)** — the world model is the architecture of
  *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from
  Pixels* (Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv:2603.19312).
  `worldmodel/wm_train.py` uses the authors' public `stable-worldmodel` /
  `stable-pretraining` packages, and the CEM evaluation follows their planning
  protocol. The pretrained cube checkpoint released with LeWM is the starting
  point for all `cube-single` experiments.
- **OGBench** (Park, Frans, Eysenbach, Levine, ICLR 2025) provides every
  environment and offline dataset used in the thesis.

---

## Setup

Experiments ran on a single A100 with one Python 3.11 venv containing **both**
stacks (PyTorch for the world model/encoder, JAX for the agents):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_frozen.txt          # exact versions used (torch 2.7 + jax 0.6 + cuda12)
# or, minimally:
pip install -r offline_to_online/requirements.txt   # JAX side
pip install torch torchvision lightning stable-worldmodel stable-pretraining ogbench h5py hdf5plugin
```

Two external pieces:

1. **Patched ogbench clone** (needed only for the latent hierarchical trainers,
   `train_hiql_acfql_latent.py` / `train_wgsp_lsg*.py`): follow
   [`external/ogbench_impls/README.md`](external/ogbench_impls/README.md) — clone
   <https://github.com/seohongpark/ogbench> to `~/ogbench`, apply the patch,
   copy the two agents in.
2. **Data home**: all scripts resolve data under `$STABLEWM_HOME`
   (default `~/stable_wm_data`), laid out as:

```
$STABLEWM_HOME/
  cube/lejepa…/                 # LeWM checkpoints (base + lejepa_play_ft_full)
  ogbench/
    visual-cube-single-play-v0.npz        # downloaded OGBench dataset
    visual-cube-single-play-v0_224.h5     # 224×224 re-render
    lewm_224_latents_cache_ftfull.pt      # per-episode 192-D latent cache
```

`MUJOCO_GL=egl` is assumed for headless rendering. The `run_*.sh` scripts in
`offline_to_online/` are SLURM wrappers around the corresponding Python entry
points; every experiment below can equally be run directly with `python`.

---

## Data and world-model pipeline

All downstream experiments consume three artefacts per environment: a 224×224
HDF5 dataset, a (fine-tuned) world-model checkpoint, and a latent cache.

```bash
# 1. download the OGBench play dataset (64×64) — any of:
#    visual-cube-single-play-v0, visual-scene-play-v0, visual-puzzle-3x3-play-v0
python -c "import ogbench; ogbench.make_env_and_datasets('visual-cube-single-play-v0')"

# 2. NPZ → HDF5 (keeps qpos/qvel, and button_states for scene/puzzle)
python data_utils/convert_ogbench_npz.py --input ... --output .../cube_single_play_v0.h5

# 3. re-render at 224×224 by replaying the physics state through MuJoCo
python data_utils/refilm_dataset.py \
    --input  $STABLEWM_HOME/ogbench/visual-cube-single-play-v0.h5 \
    --output $STABLEWM_HOME/ogbench/visual-cube-single-play-v0_224.h5 \
    --env_name visual-cube-single-v0

# 4. train the world model (Lightning + Hydra; see worldmodel/config/train/)
python worldmodel/wm_train.py --config-name lewm_ogbench
#    for cube-single we instead start from the public LeWM checkpoint and fine-tune:
python worldmodel/finetune_wm_on_play.py   # → lejepa_play_ft_full

# 5. build the latent cache with the SAME encoder the agents will use
python data_utils/reencode_play_dataset.py \
    --wm_ckpt $STABLEWM_HOME/cube/lejepa_play_ft_full/lejepa_play_ft_full \
    --hdf5    $STABLEWM_HOME/ogbench/visual-cube-single-play-v0_224.h5 \
    --out_cache $STABLEWM_HOME/ogbench/lewm_224_latents_cache_ftfull.pt
```

---

## Reproducing the evaluation chapter

Below, each section of `thesis/evaluation/evaluation.tex` is mapped to its
entry point. Defaults reproduce the headline configuration; the sweep values
are in the table captions and Appendix D/E of the thesis.

### §5.2 World-model diagnostics (drift, geometry, action conditioning)

```bash
python analysis/analyse_lewm_224.py          # 224 model: drift vs horizon, latent norms
python analysis/analyse_ogbench_wm.py        # 64×64 OGBench-trained models
python analysis/wm_ood_diagnostics.py        # action conditioning: drift under true vs
                                             # shuffled vs zero actions (+ OOD manifold)
python analysis/wm_report_diagnostics.py     # aggregate report tables/plots
python analysis/diagnose_lewm_encoder.py     # latent-space well-posedness checks
```

Run these once per environment checkpoint (`cube`, `scene`, `puzzle-3x3`) to
fill Tables 5.1–5.3; the action-conditioning ratio is the shuffled-vs-true
drift ratio from `wm_ood_diagnostics.py`.

### §5.3 CEM latent-planning baseline (Table 5.4)

```bash
python analysis/eval_cem_ogbench.py        # LeWM CEM protocol, 5 tasks × 50 episodes
```

Reports both the latent-reach rate (72.0%) and the native success rate (0.0%) —
the gap that motivates critic-grounded planning in the rest of the thesis.

### §5.4 World model as environment (Table 5.5)

All variants go through the Q-chunking `main.py`; the B1/B2/E modes (IMPALA
pixels vs. JEPA latents, real env vs. WM env) are flag presets that the
`run_*.sh` wrappers configure:

```bash
bash offline_to_online/run_b1.sh              # pure qc at 64×64 (reference pipeline)
bash offline_to_online/run_b2.sh              # control: JEPA latents + REAL env online → 100%
bash offline_to_online/run_experiment.sh      # E: online RL inside the WM (sparse reward)
bash offline_to_online/run_experiment_anchored.sh   # + latent anchoring (sparse/dense)
bash offline_to_online/run_b2_joint.sh        # joint predictor–policy training
```

Uncertainty-penalised variants are flags on the same entry point:
`--wm_ensemble_paths=...` + `--wm_uncertainty_penalty` (ensemble disagreement)
and `--wm_uncertainty_mode=knn` (nearest-neighbour regularisation). The
component-swap diagnostics load a jointly-trained predictor exported with
`worldmodel/export_jointtrained_wm.py`.

### §5.5 Inference-time MPC on a frozen policy (Tables 5.6–5.8)

```bash
python offline_to_online/eval_mpc.py \
    --policy_ckpt .../offline_final/params_500000.pkl \
    --wm_ckpt  $STABLEWM_HOME/cube/lejepa_play_ft_full/lejepa_play_ft_full \
    --wm_cache $STABLEWM_HOME/ogbench/lewm_224_latents_cache_ftfull.pt \
    --mpc_n 32 --mpc_h 1 --n_episodes 250 --task_id 1
```

- BoN-MPC: vary `--mpc_h {1,2,3}`; Q-term vs dense+Q vs Q-all scoring via
  `--mpc_q_only` / `--mpc_q_every_step`.
- Grad-MPC: add `--mpc_k_grad 5` (gradient ascent through the differentiable WM).
- FMQ: `--mpc_fmq --fmq_eta {0.01…0.5}` (single normalised trust-region step).
- `eval_bon_baseline.py` gives the planner-free best-of-N reference.

### §5.6.1–5.6.3 Interleaved learning from MPC actions (Tables 5.9–5.11)

```bash
python offline_to_online/train_interleaved_mpc.py \
    --wm_ckpt ... --wm_cache ... --task_id 1 \
    --mpc_update_mode mixed --mpc_mix_ratio 0.3 \
    --save_dir .../interleaved_fmq_slow
```

`--mpc_update_mode separate` is the actor-only setting, `mixed` the
critic-update setting (`p=0.3`). The relabel interval (25k vs 50k), label
generator (Grad-MPC vs FMQ, `η_train=0.1`), and the full Appendix D sweep are
flags on the same script. Re-running `eval_mpc.py` on the resulting 400k
checkpoint with `--fmq_eta 0.02` reproduces the 96.0% headline row.

### §5.6.4 Online actor fine-tuning with WM transitions (Table 5.12)

```bash
python offline_to_online/train_online_mpc_only.py \
    --policy_ckpt .../offline_final/params_500000.pkl \
    --wm_ckpt ... --wm_cache ... \
    --wm_consistent_targets --freeze_critic \
    --save_dir .../e2_online_frozen_s0 --seed 0
```

`--wm_consistent_targets` is Variant A (episodic imagined rollouts; omit it for
Variant B's offline-state relabelling); drop `--freeze_critic` to reproduce the
actor+critic collapse row.

### §5.6.5 Goal-conditioned online-in-WM (Table 5.13)

```bash
# Phase 1: offline GC pretraining with HER (index-based indicator reward)
python offline_to_online/train_offline_gc.py \
    --wm_ckpt ... --wm_cache ... --offline_steps 500000 \
    --save_dir $STABLEWM_HOME/cube/e3_offline_gc

# Phase 2: online actor-only fine-tuning inside the WM
python offline_to_online/train_online_mpc_gc.py \
    --policy_ckpt $STABLEWM_HOME/cube/e3_offline_gc/params_500000.pkl \
    --wm_ckpt ... --wm_cache ... \
    --online_sample_mode her --save_dir .../e3b_online_s0
```

`--online_sample_mode {her,online_only,mix}` + `--online_mix_ratio` cover the
HER/no-HER/mixed rows; `--online_goal_source {eval_goals,random_states}`
selects evaluation-goal training vs. the leakage-free achieved-state protocol,
and `--unfreeze_critic` reproduces the updated-critic row.

### §5.6.6 Static WM-generated datasets (Table 5.14)

```bash
# generate the corpora (~800k transitions each)
python offline_to_online/train_online_mpc_only.py --gen_only --bon ...   # single-task BoN
python offline_to_online/make_gc_mpc_dataset.py --mode rollout ...       # GC, imagined states + MPC actions
python offline_to_online/make_gc_mpc_dataset.py --mode relabel ...       # GC, real states + MPC actions

# train the three downstream learners on each corpus
python offline_to_online/train_bc_flow.py    --condition {original,rollout,relabel} ...
python offline_to_online/train_iql_chunk.py  --condition {original,rollout} ...
python offline_to_online/train_rebrac_chunk.py --condition {original,rollout} ...
```

(`relabel` is blocked for the value-based learners by design — its transitions
are not dynamically consistent.)

### §5.7 Generalisation to scene and puzzle-3x3 (Tables 5.15–5.18)

Run the data pipeline above with `--env_name visual-scene-v0` /
`visual-puzzle-3x3-v0` (the `button_states` field is carried through
automatically), train a fresh WM per environment, then re-run §5.6.4 / §5.6.5
with `--env_family scene` or `--env_family puzzle-3x3`. The cheap screening
diagnostic (action-conditioning ratio, Table 5.16) comes from
`analysis/wm_ood_diagnostics.py` on the new checkpoint and cache — no RL run
needed.

### §5.8 Latent Subgoal Planning (Tables 5.19–5.21, Appendix E)

Requires the patched ogbench clone (see Setup).

```bash
# Phase 1: offline hierarchical pretrain (flow HL + flow LL + IQL value) → 88.0%
python offline_to_online/train_wgsp_lsg.py \
    --save_dir ./ckpt_wgsp_lsg_p1_task12345_latent_s0 \
    --task_ids 1 2 3 4 5 --subgoal_space latent

# Phase 2: world-model-grounded HL updates from the Phase-1 checkpoint
python offline_to_online/train_wgsp_lsg_phase2.py \
    --phase1_ckpt ./ckpt_wgsp_lsg_p1_task12345_latent_s0/agent_final.pkl \
    --save_dir ./ckpt_wgsp_lsg_p2_... --task_ids 1 2 3 4 5 \
    --hl_mode awr --hl_awr_alpha 1.0 --ll_frozen
```

Thesis row ↔ `--hl_mode` mapping:

| Thesis table row | Flags |
|---|---|
| World-model AWR, α ∈ {0.1…5.0} | `--hl_mode awr --hl_awr_alpha α` |
| Offline AWR control | `--hl_mode phase1awr` |
| Hindsight Reality Forcing | `--hl_mode hrf` |
| No high-level update (control) | `--hl_mode frozen` |
| Real-candidate grounding (App. E.2) | `--hl_mode real --hl_ground_coef g --ll_mpc_coef c` |
| LL freezing / cadence / metric ablations (App. E.1) | `--ll_frozen`, `--hl_every`, `--ll_reach_metric {distance,value}` |

`eval_wgsp_lsg_planning.py` is the eval-time best-of-N gate over a frozen
Phase-1 checkpoint (the cheap test that HL×LL MPC does not beat the standalone
hierarchy); `eval_wgsp_planning.py` is the same gate for the
`train_hiql_acfql_latent.py` (HIQL+ACFQL on latents) checkpoints.

### Baselines and appendix experiments

- **HIQL end-to-end (87.0%)**: `python wgsp/train_hiql_endtoend.py`; the
  frozen-LeWM-encoder variants are `wgsp/train_hiql_baseline.py` (+ adapter
  flags) evaluated with `wgsp/eval_ogbench.py`.
- **Appendix A (SAC retrospective)**: curriculum SAC / SAC+HER / hierarchical
  SAC inside the WM live in `sac/` (`sac_train.py`, `sac_wm_train.py`,
  `sac_env_train*.py`, driven by `sac/run_all_her_experiments.sh`);
  `wgsp/train_joint.py` and `wgsp/train_joint_state.py` are the latent- and
  state-observation hierarchical variants. `sac/_smoke_*.py` are CPU smoke
  tests for the trainers.
- **Appendix B (TD-MPC2)**: run with the official
  [TD-MPC2](https://github.com/nicklashansen/tdmpc2) codebase (the vendored
  copy was removed from this repo; both pixel variants score 0.0%).
- **WGSP distillation variants** (`wgsp/train_hiql_wgsp.py`,
  `train_fmq_wgsp.py`, `train_grpo_wgsp.py`): the earlier subgoal-distillation
  line of work that preceded LSP. Not part of the final evaluation chapter,
  kept for completeness.

---

## Thesis

```bash
cd thesis && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Background figures are regenerated with `thesis/figures/make_jepa_figure.py`
and `thesis/figures/make_fql_figure.py`; the result figures are produced by the
analysis/evaluation scripts above.
