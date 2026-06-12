"""
train_grpo_wgsp.py — GRPO-on-τ=0 World-Grounded Subgoal Planning

Replaces FMQ's MLE-on-synthetic-target (which collapsed when the LL distribution
narrowed under HIQL anchor: log π(a_star) → -∞, gradient → ∞) with group-
relative policy optimisation on real-state actions only.

  * HL: soft AWR (cross-N candidate advantage, softmax-weighted MLE).
  * LL: soft AWR on the FIRST action of each WM rollout (τ=0), so the MLE
        target is an action actually sampled from the current policy at the
        real state z_t. Within-rep advantage A^{n,m} = J^{n,m} - mean_m J,
        softmax weights u^{n,m}, cross-rep weighting w^n from HL.
        L_GRPO = - Σ_n Σ_m  w^n · u^{n,m} · log π^L(a_0^{n,m} | z_t, rep^n)
  * V : real data only (unchanged).

Why this fixes the FMQ collapse:
  - MLE targets are IN-DISTRIBUTION (sampled from π itself) → log_prob
    bounded by construction → no gradient explosion.
  - WM enters only as a SCALAR scorer (J = V(z_k, rep)); its gradient is not
    used. Robust to bad WM gradient geometry (the original WGSP failure).
  - Implicit trust region via softmax weights ∈ [0,1].

Why τ=0 only (not the whole rollout as in original WGSP-distil):
  - τ=0 actions are sampled at REAL z_t.
  - τ≥1 actions are sampled at IMAGINED z_τ; using them creates the same
    self-amplifying drift loop that killed the original WGSP-distil. Slicing
    to τ=0 breaks the loop while keeping the in-distribution guarantee.
"""

import sys
import os

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import argparse
import csv
import time

import numpy as np
import torch
from torch.distributions import Normal

import stable_worldmodel as swm
from hydra import initialize, compose

from train_action_decoder import ActionChunkDecoder
from train_hiql_wgsp import (
    GoalRep, GaussianActor, EnsembleValue, LatentAdapter, TwinValue,
    RealOfflineCache, _value_loss, _wm_predict,
    _score_endpoints, hl_wgsp_step,
    ll_hiql_loss, hl_hiql_loss, _load_jepa_from_ckpt,
    _load_legacy_twin_value, FlowDecoderWrapper,
)


# =============================================================================
# GRPO-on-τ=0 LL update
# =============================================================================

def ll_grpo_step(rollouts, ll_actor, adapter, return_diag=False):
    """LL update via group-relative policy optimisation on τ=0 actions.

    Uses the within-rep softmax weights u_ll already computed in
    hl_wgsp_step (from endpoint J values, cross-M baseline). Cross-rep
    weighting via w_hl prioritises actions taken under HL-preferred reps.

    Restricting to τ=0 means MLE targets are actions sampled at the REAL
    state z_t (not imagined rollout states), so the in-distribution log_prob
    guarantee holds AND no self-amplifying drift through imagined states.

    Args:
        rollouts: dict from hl_wgsp_step(..., return_rollouts=True). Must
                  carry: traj_z [BNM, k+1, 192], traj_a_pre [BNM, k, adim],
                         rep_NM [BNM, rep_dim], w_hl [B, N], u_ll [B, N, M],
                         B, N, M, k.

    Returns:
        grpo_loss: scalar loss tensor (graph attached on LL params).
        diag (optional dict): log_p_mean, weight_max, weight_entropy.
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    B, N, M = rollouts['B'], rollouts['N'], rollouts['M']
    BNM = B * N * M

    traj_z     = rollouts['traj_z']        # [BNM, k+1, 192] raw WM latents
    traj_a_pre = rollouts['traj_a_pre']    # [BNM, k, adim] pre-tanh samples
    rep_NM     = rollouts['rep_NM']        # [BNM, rep_dim]
    w_hl       = rollouts['w_hl']          # [B, N]
    u_ll       = rollouts['u_ll']          # [B, N, M]

    # τ=0 slice: real anchor + first sampled action only.
    z_t_raw = traj_z[:, 0, :]              # [BNM, 192]
    a_0     = traj_a_pre[:, 0, :]          # [BNM, adim] pre-tanh
    z_t_pol = adapt(z_t_raw)               # [BNM, POLICY_DIM]

    # log π(a_0 | z_t, rep) in pre-tanh sample coords (matches sample() path).
    mean, log_std = ll_actor(z_t_pol, rep_NM)
    std = log_std.exp()
    log_p = Normal(mean, std).log_prob(a_0).sum(dim=-1)   # [BNM]

    # Tanh-squash Jacobian correction so we recover log π^L(a_post).
    if ll_actor.tanh_squash:
        y = torch.tanh(a_0)
        lp_jac = torch.log(
            ll_actor.action_scale * (1.0 - y.pow(2)) + 1e-6).sum(dim=-1)
        log_p = log_p - lp_jac

    log_p = log_p.reshape(B, N, M)

    # Two-level weight: w_hl^(n) · u_ll^(n,m). Both already detached.
    weight = (w_hl.unsqueeze(-1) * u_ll)   # [B, N, M]
    grpo_loss = -(weight * log_p).sum(dim=(1, 2)).mean()

    if return_diag:
        diag = {
            'log_p_mean':     log_p.mean().item(),
            'weight_max':     weight.amax(dim=(1, 2)).mean().item(),
            'weight_entropy': -(weight * (weight + 1e-12).log()).sum(
                dim=(1, 2)).mean().item(),
        }
        return grpo_loss, diag
    return grpo_loss, None


# =============================================================================
# Training loop
# =============================================================================

def grpo_train_loop(
    wm_model,
    hl_actor,
    ll_actor,
    value_net,
    value_target,
    goal_rep, goal_rep_target,
    real_cache,
    decoder,
    optimizer,
    trainable_params,
    use_wgsp, use_grpo, use_decoder, use_geometric_term, use_v_in_J,
    total_steps, warmup_fraction,
    subgoal_steps, k_plan,
    N, M,
    batch_size,
    gamma, tau, expectile,
    alpha_H, alpha_L,
    alpha_H_wgsp, alpha_L_grpo,
    beta_geom, lambda_anchor,
    action_scale, rep_dim,
    save_dir, device,
    lambda_grpo=1.0, lambda_hiql_ll=1.0,
    log_interval=100, save_interval=10000,
    lambda_mopo=0.0,
    adapter=None,
    hl_wgsp_coeff=10.0,
    lambda_hiql_reg=0.1,
):
    os.makedirs(save_dir, exist_ok=True)
    warmup_steps = int(total_steps * warmup_fraction)
    use_5d = use_decoder   # LL outputs 5D when decoder is active

    print(f"\n{'='*65}", flush=True)
    print(f"  GRPO-WGSP training (τ=0 GRPO, M={M} rollouts/cand)", flush=True)
    print(f"  use_wgsp           : {use_wgsp}", flush=True)
    print(f"  use_grpo           : {use_grpo}  (λ_grpo={lambda_grpo}, α_L={alpha_L_grpo})", flush=True)
    print(f"  use_decoder (LL 5D): {use_decoder}", flush=True)
    print(f"  use_geometric_term : {use_geometric_term}  (β={beta_geom})", flush=True)
    print(f"  N candidates       : {N}", flush=True)
    print(f"  M rollouts/cand    : {M}", flush=True)
    print(f"  k_plan             : {k_plan}", flush=True)
    print(f"  warmup_steps       : {warmup_steps:,}  (wf={warmup_fraction})", flush=True)
    print(f"  total_steps        : {total_steps:,}", flush=True)
    print(f"  λ_anchor           : {lambda_anchor}", flush=True)
    print(f"  λ_hiql_ll          : {lambda_hiql_ll}", flush=True)
    print(f"  λ_hiql_reg (HL)    : {lambda_hiql_reg}", flush=True)
    print(f"  hl_wgsp_coeff      : {hl_wgsp_coeff}", flush=True)
    if adapter is not None:
        in_d  = adapter.mlp[0].in_features
        out_d = next(l.out_features for l in reversed(list(adapter.mlp))
                     if hasattr(l, 'out_features'))
        n_p   = sum(p.numel() for p in adapter.parameters())
        print(f"  adapter (uniform)  : {in_d}D → {out_d}D  ({n_p:,} params)", flush=True)
    print(f"{'='*65}\n", flush=True)

    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'step', 'phase', 'value_loss',
            'll_hiql_loss', 'll_grpo_loss',
            'hl_hiql_loss', 'hl_wgsp_loss',
            'grpo_log_p', 'grpo_w_max', 'grpo_w_entropy',
            'elapsed_s',
            'v_std_endpoint', 'v_std_data',
            'mu_norm', 'mu_target_cos', 'rep_cand_pair_dist',
            'z_k_std_NM', 'z_k_disp', 'closer_frac',
            'v_mean_range_N', 'J_std_N',
            'w_hl_max', 'w_hl_entropy',
        ])

    t0 = time.time()
    t_int = time.time()
    sum_v = sum_ll_hiql = sum_ll_grpo = 0.0
    sum_hl_hiql = sum_hl_wgsp = 0.0
    sum_grpo_logp = sum_grpo_wmax = sum_grpo_went = 0.0
    sum_v_std_ep = sum_v_std_dat = 0.0
    sum_mu_norm = sum_rep_pair = sum_mu_tgt_cos = 0.0
    sum_z_k_disp = sum_closer = sum_z_k_std = 0.0
    sum_v_mean_range = sum_J_std_N = 0.0
    sum_w_hl_max = sum_w_hl_ent = 0.0
    audit_log_n = diag_log_n = grpo_log_n = log_n = 0
    prev_phase = 1

    for step in range(total_steps):
        phase = 1 if step < warmup_steps else 2
        if phase != prev_phase:
            print(f"\n>>> Phase 2 begins at step {step:,} — WGSP+GRPO active\n",
                  flush=True)
            prev_phase = phase

        # ── Value loss (sparse HER, real data only) ──────────────────────────
        z_hl, z_next_hl, g_her, success_hl = real_cache.sample_value_her_batch(
            batch_size)
        r_hl    = success_hl - 1.0
        mask_hl = 1.0 - success_hl
        v_loss = _value_loss(
            value_net, value_target, goal_rep, goal_rep_target,
            z_hl, z_next_hl, g_her, r_hl, gamma, expectile,
            adapter=adapter, mask=mask_hl)
        sum_v += v_loss.item()

        # ── LL HIQL (always on; real-data anchor) ────────────────────────────
        ll_loss = ll_hiql_loss(real_cache, ll_actor, value_net, goal_rep,
                               batch_size, subgoal_steps, alpha_L,
                               use_5d=use_5d, mode='awr', adapter=adapter)
        sum_ll_hiql += ll_loss.item()

        # ── HL: WGSP or HIQL ─────────────────────────────────────────────────
        hl_loss = None
        hl_hiql_reg = None
        ll_grpo = None

        if use_wgsp and phase == 2:
            z_t, z_target, g_ult = real_cache.sample_hl_batch(
                batch_size, subgoal_steps)

            hl_loss, rollouts = hl_wgsp_step(
                z_t=z_t, g_ult=g_ult,
                hl_actor=hl_actor, ll_actor=ll_actor, decoder=decoder,
                wm_model=wm_model, goal_rep=goal_rep, value_net=value_net,
                N=N, M=M, k=k_plan,
                beta_geom=beta_geom,
                alpha_H=alpha_H_wgsp,
                alpha_L=alpha_L_grpo,          # within-rep AWR temperature for u_ll
                action_scale=action_scale,
                use_decoder=use_decoder,
                use_geometric_term=use_geometric_term,
                use_v_in_J=use_v_in_J,
                lambda_anchor=lambda_anchor, z_target=z_target,
                return_rollouts=True,          # rollouts feed GRPO LL update
                lambda_mopo=lambda_mopo,
                return_v_diag=True,
                ll_score_mode='goal',          # J_ll = V-based J (not rep_reach)
                adapter=adapter,
                value_target=value_target, goal_rep_target=goal_rep_target,
            )
            sum_hl_wgsp += hl_loss.item()

            # HIQL HL regulariser (prevents μ-norm drift)
            hl_hiql_reg = hl_hiql_loss(
                real_cache, hl_actor, value_net, goal_rep,
                batch_size, subgoal_steps, alpha_H, adapter=adapter)
            sum_hl_hiql += hl_hiql_reg.item()

            # Collect diagnostics from rollouts
            if rollouts is not None:
                vd = rollouts.get('v_diag')
                if vd is not None:
                    if vd.get('v_std') is not None:
                        sum_v_std_ep += vd['v_std'].mean().item()
                        with torch.no_grad():
                            z_dat, _, _, _, z_sub_dat = real_cache.sample_ll_batch(
                                batch_size, subgoal_steps, use_5d=False)
                            _adapt = (lambda x: adapter(x)) if adapter is not None \
                                     else (lambda x: x)
                            rep_dat = goal_rep_target(
                                _adapt(z_dat), _adapt(z_sub_dat))
                            v_dat_all = value_target(_adapt(z_dat), rep_dat)
                            v_std_dat = (v_dat_all.std(dim=-1, unbiased=True)
                                         if v_dat_all.shape[-1] > 1
                                         else torch.zeros(batch_size, device=device))
                        sum_v_std_dat += v_std_dat.mean().item()
                        audit_log_n += 1
                    if vd.get('J_per_rep_std_N') is not None:
                        sum_mu_norm   += vd['mu_norm'].mean().item()
                        sum_rep_pair  += vd['rep_cand_pair_dist'].mean().item()
                        if vd.get('mu_target_cos') is not None:
                            sum_mu_tgt_cos += vd['mu_target_cos'].mean().item()
                        sum_z_k_std   += vd['z_k_std_NM'].mean().item()
                        sum_z_k_disp  += vd['z_k_disp'].mean().item()
                        sum_closer    += vd['closer_frac'].mean().item()
                        if vd.get('v_mean_range_N') is not None:
                            sum_v_mean_range += vd['v_mean_range_N'].mean().item()
                        sum_J_std_N   += vd['J_per_rep_std_N'].mean().item()
                        sum_w_hl_max  += vd['w_hl_max'].mean().item()
                        sum_w_hl_ent  += vd['w_hl_entropy'].mean().item()
                        diag_log_n    += 1

            # ── GRPO LL step (τ=0 actions, in-distribution MLE) ──────────────
            if use_grpo and rollouts is not None:
                ll_grpo, grpo_diag = ll_grpo_step(
                    rollouts=rollouts, ll_actor=ll_actor, adapter=adapter,
                    return_diag=True,
                )
                sum_ll_grpo += ll_grpo.item()
                if grpo_diag is not None:
                    sum_grpo_logp += grpo_diag['log_p_mean']
                    sum_grpo_wmax += grpo_diag['weight_max']
                    sum_grpo_went += grpo_diag['weight_entropy']
                    grpo_log_n += 1

        else:
            # Warmup or no WGSP: pure HIQL HL
            hl_loss = hl_hiql_loss(
                real_cache, hl_actor, value_net, goal_rep,
                batch_size, subgoal_steps, alpha_H, adapter=adapter)
            sum_hl_hiql += hl_loss.item()

        # ── Combined backward ─────────────────────────────────────────────────
        total_loss = v_loss + lambda_hiql_ll * ll_loss
        if hl_loss is not None:
            if use_wgsp and phase == 2:
                total_loss = total_loss + hl_wgsp_coeff * hl_loss
            else:
                total_loss = total_loss + hl_loss
        if hl_hiql_reg is not None:
            total_loss = total_loss + lambda_hiql_reg * hl_hiql_reg
        if ll_grpo is not None:
            total_loss = total_loss + lambda_grpo * ll_grpo

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        # Soft-update target nets
        with torch.no_grad():
            for tp, p in zip(value_target.parameters(), value_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)
            for tp, p in zip(goal_rep_target.parameters(), goal_rep.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)

        log_n += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_interval == 0:
            d   = max(log_n, 1)
            ad  = max(audit_log_n, 1)
            dd  = max(diag_log_n, 1)
            gd  = max(grpo_log_n, 1)
            now = time.time()
            sps = log_interval / max(now - t_int, 1e-6)
            remaining = (total_steps - step) / max(sps, 1e-6)
            elapsed = now - t0

            print(
                f"Step {step:06d}/{total_steps:,} ({100.*step/total_steps:4.1f}%) "
                f"| Ph{phase} | "
                f"V {sum_v/(2*d):.4f} | "
                f"LL_h {sum_ll_hiql/d:.4f} LL_grpo {sum_ll_grpo/d:.4f} | "
                f"HL_h {sum_hl_hiql/d:.4f} HL_w {sum_hl_wgsp/d:.4f} | "
                f"GRPO lp {sum_grpo_logp/gd:.2f} wmax {sum_grpo_wmax/gd:.3f} "
                f"wH {sum_grpo_went/gd:.3f} | "
                f"Vstd ep/dat {sum_v_std_ep/ad:.3f}/{sum_v_std_dat/ad:.3f} | "
                f"closer {sum_closer/dd:.3f} Jσ {sum_J_std_N/dd:.3f} | "
                f"HLwmax {sum_w_hl_max/dd:.3f} | "
                f"{sps:.1f} sps | ETA {remaining/60:.0f}m",
                flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, phase, sum_v/(2*d),
                    sum_ll_hiql/d, sum_ll_grpo/d,
                    sum_hl_hiql/d, sum_hl_wgsp/d,
                    sum_grpo_logp/gd, sum_grpo_wmax/gd, sum_grpo_went/gd,
                    now - t_int,
                    sum_v_std_ep/ad, sum_v_std_dat/ad,
                    sum_mu_norm/dd, sum_mu_tgt_cos/dd, sum_rep_pair/dd,
                    sum_z_k_std/dd, sum_z_k_disp/dd, sum_closer/dd,
                    sum_v_mean_range/dd, sum_J_std_N/dd,
                    sum_w_hl_max/dd, sum_w_hl_ent/dd,
                ])
            # Reset accumulators
            sum_v = sum_ll_hiql = sum_ll_grpo = 0.0
            sum_hl_hiql = sum_hl_wgsp = 0.0
            sum_grpo_logp = sum_grpo_wmax = sum_grpo_went = 0.0
            sum_v_std_ep = sum_v_std_dat = 0.0
            sum_mu_norm = sum_rep_pair = sum_mu_tgt_cos = 0.0
            sum_z_k_disp = sum_closer = sum_z_k_std = 0.0
            sum_v_mean_range = sum_J_std_N = 0.0
            sum_w_hl_max = sum_w_hl_ent = 0.0
            audit_log_n = diag_log_n = grpo_log_n = log_n = 0
            t_int = now

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step > 0 and (step % save_interval == 0 or step == total_steps - 1):
            torch.save(ll_actor.state_dict(),  os.path.join(save_dir, 'll_actor.pth'))
            torch.save(hl_actor.state_dict(),  os.path.join(save_dir, 'hl_actor.pth'))
            torch.save(value_net.state_dict(), os.path.join(save_dir, 'value_net.pth'))
            torch.save(goal_rep.state_dict(),  os.path.join(save_dir, 'goal_rep.pth'))
            if adapter is not None:
                torch.save(adapter.state_dict(),
                           os.path.join(save_dir, 'adapter.pth'))
            import json as _json
            with open(os.path.join(save_dir, 'metadata.json'), 'w') as _mf:
                _json.dump({'encode_mode': 'cls_projected'}, _mf)
            print(f"  → checkpoint saved at step {step}", flush=True)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    def _bool(s):
        return str(s).lower() in ('1', 'true', 'yes', 'y', 't')

    parser = argparse.ArgumentParser(description='GRPO-WGSP training')

    # Paths
    parser.add_argument('--ckpt_path',    type=str, default=None)
    parser.add_argument('--cache_path',   type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--save_dir',     type=str, default=None)
    parser.add_argument('--decoder_ckpt', type=str, default=None)
    parser.add_argument('--decoder_type', type=str, default='mse',
                        choices=['mse', 'flow'])

    # Schedule
    parser.add_argument('--total_steps',     type=int,   default=200_000)
    parser.add_argument('--warmup_fraction', type=float, default=0.1)
    parser.add_argument('--subgoal_steps',   type=int,   default=8)
    parser.add_argument('--k_plan',          type=int,   default=None)
    parser.add_argument('--N',               type=int,   default=8)
    parser.add_argument('--M',               type=int,   default=4,
                        help='Rollouts per candidate for HL scoring and '
                             'within-rep advantage signal for GRPO LL.')
    parser.add_argument('--batch_size',      type=int,   default=256)

    # IQL / AWR temperatures
    parser.add_argument('--alpha_H',         type=float, default=3.0)
    parser.add_argument('--alpha_L',         type=float, default=3.0)
    parser.add_argument('--alpha_H_wgsp',    type=float, default=3.0)
    parser.add_argument('--alpha_L_grpo',    type=float, default=3.0,
                        help='Within-rep AWR temperature for GRPO u_ll '
                             'softmax weights.')
    parser.add_argument('--beta_geom',       type=float, default=0.0)
    parser.add_argument('--lambda_anchor',   type=float, default=0.0)

    # GRPO-specific
    parser.add_argument('--lambda_grpo',     type=float, default=1.0,
                        help='Weight on GRPO LL loss.')
    parser.add_argument('--lambda_hiql_ll',  type=float, default=1.0,
                        help='Weight on HIQL LL anchor loss.')

    # RL params
    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--expectile',     type=float, default=0.7)
    parser.add_argument('--action_scale',  type=float, default=3.0)
    parser.add_argument('--rep_dim',       type=int,   default=10)

    # Toggles
    parser.add_argument('--use_wgsp',           type=_bool, default=True)
    parser.add_argument('--use_grpo',           type=_bool, default=True)
    parser.add_argument('--use_decoder',        type=_bool, default=True)
    parser.add_argument('--use_geometric_term', type=_bool, default=False)
    parser.add_argument('--use_v_in_J',         type=_bool, default=True)

    # Encoder
    parser.add_argument('--img_size',   type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--seed',       type=int, default=0)

    # Adapter
    parser.add_argument('--use_adapter',        type=_bool, default=True)
    parser.add_argument('--adapter_dim',        type=int,   default=256)
    parser.add_argument('--adapter_hidden_dim', type=int,   default=None)
    parser.add_argument('--adapter_depth',      type=int,   default=2)

    # Loss coefficients
    parser.add_argument('--hl_wgsp_coeff',   type=float, default=10.0)
    parser.add_argument('--lambda_hiql_reg', type=float, default=0.1)
    parser.add_argument('--lambda_mopo',     type=float, default=0.0)
    parser.add_argument('--n_value_heads',   type=int,   default=2)

    args = parser.parse_args()

    if args.k_plan is None:
        args.k_plan = args.subgoal_steps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    data_path  = args.dataset_path or os.path.join(
        STABLEWM_HOME, 'ogbench', 'visual-cube-single-play-v0_224.h5')
    cache_path = args.cache_path or os.path.join(
        STABLEWM_HOME, 'lewm_224_latents_cache.pt')
    _adapter_tag = f'_adapter{args.adapter_dim}' if args.use_adapter else '_noadapter'
    save_dir = args.save_dir or (
        f'./checkpoints_grpo_wgsp_k{args.subgoal_steps}_N{args.N}_M{args.M}'
        f'_aLg{args.alpha_L_grpo}_lgrpo{args.lambda_grpo}'
        f'{_adapter_tag}_s{args.seed}')

    # ── Load encoder / WM ────────────────────────────────────────────────────
    # Always use load_jepa from jepa_loader: it adds leworldmodel/ to sys.path
    # (needed for 'jepa' module when unpickling), tries {path}_state_dict.pt
    # first (no CostShim pickle issues), and falls back to AutoCostModel.
    _o2o_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'offline_to_online'))
    if _o2o_dir not in sys.path:
        sys.path.append(_o2o_dir)
    from envs.jepa_loader import load_jepa

    if args.ckpt_path is not None:
        _wm_ckpt_path = args.ckpt_path
    else:
        # Resolve default WM from hydra config (cfg.policy = 'cube/lejepa',
        # relative to STABLEWM_HOME).
        with initialize(version_base=None, config_path='../../config'):
            cfg = compose(config_name='eval/cube',
                          overrides=['+policy=cube/lejepa'])
        _stablewm = os.environ.get('STABLEWM_HOME',
                                   os.path.expanduser('~/stable_wm_data'))
        _wm_ckpt_path = os.path.join(_stablewm, cfg.policy)

    wm_model = load_jepa(
        _wm_ckpt_path, device=str(device),
        img_size=args.img_size, patch_size=args.patch_size)
    for p in wm_model.parameters():
        p.requires_grad = False

    # ── Load latent cache + action scaler ────────────────────────────────────
    print(f'Loading cache from {cache_path} ...')
    cache_data  = torch.load(cache_path, map_location='cpu')
    all_latents = cache_data['all_latents']
    all_actions = cache_data.get('all_actions', [])
    if not all_actions:
        # Cache was built without actions (e.g. ftfull cache only stores latents).
        # Reconstruct per-episode action lists from the HDF5 dataset.
        print('Cache has no all_actions — loading from dataset HDF5 ...')
        import h5py
        with h5py.File(data_path, 'r') as _hf:
            _acts   = torch.tensor(np.array(_hf['action']),    dtype=torch.float32)
            _offsets = np.array(_hf['ep_offset'])
            _lengths = np.array(_hf['ep_len'])
        all_actions = [
            _acts[int(_offsets[i]): int(_offsets[i]) + int(_lengths[i])]
            for i in range(len(_offsets))
        ]
        print(f'  Loaded {len(all_actions)} episodes from HDF5.')

    from sklearn import preprocessing as sk_pre
    print('Fitting StandardScaler on dataset actions ...')
    _dp_stem = data_path.rstrip('/')
    if _dp_stem.endswith('.h5'):
        _dp_stem = _dp_stem[:-3]
    import stable_worldmodel as _swm
    _ds_for_scaler = _swm.data.HDF5Dataset(
        _dp_stem, keys_to_cache=['action'],
        cache_dir=os.path.dirname(_dp_stem))
    action_raw = _ds_for_scaler.get_col_data('action')
    action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(action_raw)
    print(f'  Scaler fit on {len(action_raw):,} action frames')

    real_cache = RealOfflineCache(
        all_latents=all_latents, all_actions=all_actions,
        action_scaler=action_scaler, device=device, frameskip=5)

    # ── Network setup ─────────────────────────────────────────────────────────
    LATENT_DIM  = 192
    REP_DIM     = args.rep_dim
    HIDDEN_DIMS = (512, 512, 512)
    LL_OUT_DIM  = 5 if args.use_decoder else 25

    adapter = None
    POLICY_DIM = LATENT_DIM
    if args.use_adapter:
        adapter = LatentAdapter(
            in_dim=LATENT_DIM, out_dim=args.adapter_dim,
            hidden_dim=args.adapter_hidden_dim,
            depth=args.adapter_depth).to(device)
        POLICY_DIM = args.adapter_dim
        n_a = sum(p.numel() for p in adapter.parameters())
        print(f'LatentAdapter: {LATENT_DIM}D → {POLICY_DIM}D  ({n_a:,} params)')

    goal_rep = GoalRep(latent_dim=POLICY_DIM, rep_dim=REP_DIM,
                       hidden_dims=HIDDEN_DIMS).to(device)
    goal_rep_target = GoalRep(latent_dim=POLICY_DIM, rep_dim=REP_DIM,
                              hidden_dims=HIDDEN_DIMS).to(device)
    goal_rep_target.load_state_dict(goal_rep.state_dict())
    for p in goal_rep_target.parameters():
        p.requires_grad = False

    ll_actor = GaussianActor(
        state_dim=POLICY_DIM, goal_dim=REP_DIM, output_dim=LL_OUT_DIM,
        hidden_dims=HIDDEN_DIMS, tanh_squash=True,
        action_scale=args.action_scale).to(device)

    hl_actor = GaussianActor(
        state_dim=POLICY_DIM, goal_dim=POLICY_DIM, output_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, tanh_squash=False, const_std=True).to(device)

    value_net = EnsembleValue(
        latent_dim=POLICY_DIM, rep_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, n_heads=args.n_value_heads).to(device)
    value_target = EnsembleValue(
        latent_dim=POLICY_DIM, rep_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, n_heads=args.n_value_heads).to(device)
    value_target.load_state_dict(value_net.state_dict())
    for p in value_target.parameters():
        p.requires_grad = False

    # ── Decoder ───────────────────────────────────────────────────────────────
    decoder = None
    if args.use_decoder:
        if args.decoder_ckpt is None:
            raise ValueError('--decoder_ckpt required when --use_decoder=True')
        if args.decoder_type == 'flow':
            from train_flow_action_decoder import FlowChunkDecoder
            meta_path = os.path.join(
                os.path.dirname(args.decoder_ckpt), 'flow_decoder_meta.pt')
            flow_steps = 10
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location='cpu')
                flow_steps = meta.get('flow_steps', 10)
            _flow = FlowChunkDecoder(
                latent_dim=LATENT_DIM, a_first_dim=5, action_dim=25,
                hidden_dims=(512, 512, 512, 512)).to(device)
            _flow.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
            _flow.eval()
            for p in _flow.parameters():
                p.requires_grad = False
            decoder = FlowDecoderWrapper(_flow, flow_steps=flow_steps).to(device)
        else:
            _dec_sd = torch.load(args.decoder_ckpt, map_location='cpu')
            _goal_dim = max(0, _dec_sd['net.0.weight'].shape[1] - 5 - LATENT_DIM)
            decoder = ActionChunkDecoder(
                in_dim=5, out_dim=25, latent_dim=LATENT_DIM,
                hidden_dims=(256, 256), goal_dim=_goal_dim).to(device)
            decoder.load_state_dict(_dec_sd)
            decoder.eval()
            for p in decoder.parameters():
                p.requires_grad = False
            print(f'  Loaded frozen ActionChunkDecoder from {args.decoder_ckpt} '
                  f'(goal_dim={_goal_dim})')

    total_params = (
        sum(p.numel() for p in goal_rep.parameters())
        + sum(p.numel() for p in ll_actor.parameters())
        + sum(p.numel() for p in hl_actor.parameters())
        + sum(p.numel() for p in value_net.parameters()))
    print(f'Trainable params: {total_params:,}  '
          f'(POLICY_DIM={POLICY_DIM}, REP_DIM={REP_DIM}, '
          f'LL_OUT_DIM={LL_OUT_DIM})')

    adapter_params = list(adapter.parameters()) if adapter is not None else []
    trainable_params = (
        adapter_params
        + list(goal_rep.parameters())
        + list(value_net.parameters())
        + list(ll_actor.parameters())
        + list(hl_actor.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=3e-4)

    grpo_train_loop(
        wm_model=wm_model,
        hl_actor=hl_actor, ll_actor=ll_actor,
        value_net=value_net, value_target=value_target,
        goal_rep=goal_rep, goal_rep_target=goal_rep_target,
        real_cache=real_cache, decoder=decoder,
        optimizer=optimizer, trainable_params=trainable_params,
        use_wgsp=args.use_wgsp, use_grpo=args.use_grpo,
        use_decoder=args.use_decoder,
        use_geometric_term=args.use_geometric_term,
        use_v_in_J=args.use_v_in_J,
        total_steps=args.total_steps,
        warmup_fraction=args.warmup_fraction,
        subgoal_steps=args.subgoal_steps, k_plan=args.k_plan,
        N=args.N, M=args.M, batch_size=args.batch_size,
        gamma=args.gamma, tau=0.005, expectile=args.expectile,
        alpha_H=args.alpha_H, alpha_L=args.alpha_L,
        alpha_H_wgsp=args.alpha_H_wgsp, alpha_L_grpo=args.alpha_L_grpo,
        beta_geom=args.beta_geom, lambda_anchor=args.lambda_anchor,
        action_scale=args.action_scale, rep_dim=REP_DIM,
        save_dir=save_dir, device=device,
        lambda_grpo=args.lambda_grpo,
        lambda_hiql_ll=args.lambda_hiql_ll,
        lambda_mopo=args.lambda_mopo,
        adapter=adapter,
        hl_wgsp_coeff=args.hl_wgsp_coeff,
        lambda_hiql_reg=args.lambda_hiql_reg,
    )
