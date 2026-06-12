"""
train_fmq_wgsp.py — FMQ-Grounded Subgoal Planning

Replaces WGSP's LL distillation (which caused collapse via a self-amplifying
loop through imagined states) with Flow Map Q-Guidance (FMQ):

  * HL: identical to WGSP — soft AWR (all N candidates, advantage-normed).
  * LL: FMQ trust-region refinement at REAL states z_t.
        a_0* = clip(a_0 + η · ∇_{a_0} J_LL / ‖∇_{a_0} J_LL‖, -as, as)
        L_FMQ = -log π^L(a_0* | z_t, rep^(i*))
  * V : trained on real data only (unchanged).
  * No LL distil loop → no self-amplifying collapse possible.

Design guarantees (from mpc-distill.tex analysis):
  - WM only used for ranking candidates (HL) and direction-finding (LL FMQ).
  - WM never appears in V's training signal.
  - LL trained at real z_t with FMQ-improved action labels — not imagined states.
  - No V mixed-batch injection needed: V(z, rep) has no action input, so
    there is no Q(z,a) to "calibrate on FMQ actions."

Reference:
  FMQ trust-region: a_0* = argmax_{‖u-u_ref‖≤η} Q_lin(u)
  → closed-form solution: u* = u_ref + η · ∇Q/‖∇Q‖  (Theorem 3.2 of
    Ziakas et al., "Aligning Flow Maps with Q-Guidance", 2026)
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
# FMQ LL refinement step
# =============================================================================

def ll_fmq_step(z_t, g_ult, rep_star,
                ll_actor, decoder, wm_model, value_net,
                adapter, action_scale, eta_fmq, use_decoder=True,
                return_diag=False):
    """FMQ trust-region LL label at real z_t.

    Computes the unique optimal first action (within the η-ball) toward
    rep_star, then trains the LL to produce that action.

    Args:
        z_t      [B, 192]   real WM latents from buffer
        g_ult    [B, 192]   raw goal latents (for goal-conditioned decoder)
        rep_star [B, rep_d] best candidate rep from HL (detached)

    Returns:
        fmq_loss: scalar MLE loss for LL
        diag (optional dict): grad_norm, J_LL_mean, a_displacement
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)

    # Policy-space inputs, detached so FMQ backward doesn't touch adapter params.
    z_t_pol    = adapt(z_t).detach()       # [B, POLICY_DIM]
    rep_star_d = rep_star.detach()

    # Sample a_0 from LL in post-tanh space, detach from LL params.
    with torch.no_grad():
        a_0_samp, _, _ = ll_actor.sample(z_t_pol, rep_star_d)
    # Enable grad on a_0 so ∇_{a_0} J_LL can be computed.
    a_0 = a_0_samp.detach().requires_grad_(True)   # [B, 5]

    # H_LL = 1 WM step (most reliable FMQ gradient; mpc-distill confirmed H=1
    # minimises compounding WM error in the gradient chain).
    z_t_dg = z_t.detach()
    g_dg   = g_ult.detach()

    if use_decoder:
        goal_arg = g_dg if (hasattr(decoder, 'goal_dim')
                            and getattr(decoder, 'goal_dim', 0) > 0) else None
        c_0 = decoder(a_0, z_t_dg, goal_arg)   # [B,25]; grad: a_0 → c_0
    else:
        c_0 = a_0                                # 25D LL directly

    # WM step: grad flows c_0 → z_1 → J_LL → a_0
    z_1_raw = _wm_predict(wm_model, z_t_dg, c_0)   # [B, 192]
    z_1_pol = adapt(z_1_raw)                         # [B, POLICY_DIM]

    # Score endpoint w.r.t. subgoal (not ultimate goal — LL pursues rep_star).
    J_LL = value_net(z_1_pol, rep_star_d).mean(dim=-1)   # [B]

    # FMQ gradient: closed-form optimum of trust-region QP.
    g = torch.autograd.grad(J_LL.sum(), a_0, create_graph=False)[0]   # [B, 5]

    g_norm = g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a_star = (a_0 + eta_fmq * g / g_norm).clamp(
        -action_scale, action_scale).detach()   # [B, 5]

    # LL update: MLE on FMQ-improved label at real z_t.
    # Gradients flow into LL actor params only (z_t_pol and rep_star_d detached).
    fmq_loss = -ll_actor.log_prob(z_t_pol, rep_star_d, a_star).mean()

    if return_diag:
        diag = {
            'grad_norm':     g_norm.squeeze(-1).mean().item(),
            'J_LL_mean':     J_LL.mean().item(),
            'a_displacement': (a_star - a_0_samp).norm(dim=-1).mean().item(),
        }
        return fmq_loss, diag
    return fmq_loss, None


# =============================================================================
# Training loop
# =============================================================================

def fmq_train_loop(
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
    use_wgsp, use_fmq, use_decoder, use_geometric_term, use_v_in_J,
    total_steps, warmup_fraction,
    subgoal_steps, k_plan,
    N, M,
    batch_size,
    gamma, tau, expectile,
    alpha_H, alpha_L,
    alpha_H_wgsp,
    beta_geom, lambda_anchor,
    action_scale, rep_dim,
    save_dir, device,
    eta_fmq=0.1, lambda_fmq=1.0, lambda_hiql_ll=1.0,
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
    print(f"  FMQ-WGSP training", flush=True)
    print(f"  use_wgsp           : {use_wgsp}", flush=True)
    print(f"  use_fmq            : {use_fmq}  (η={eta_fmq}, λ={lambda_fmq})", flush=True)
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
            'll_hiql_loss', 'll_fmq_loss',
            'hl_hiql_loss', 'hl_wgsp_loss',
            'fmq_grad_norm', 'fmq_J_LL_mean', 'fmq_a_disp',
            'elapsed_s',
            'v_std_endpoint', 'v_std_data',
            'mu_norm', 'mu_target_cos', 'rep_cand_pair_dist',
            'z_k_std_NM', 'z_k_disp', 'closer_frac',
            'v_mean_range_N', 'J_std_N',
            'w_hl_max', 'w_hl_entropy',
        ])

    t0 = time.time()
    t_int = time.time()
    sum_v = sum_ll_hiql = sum_ll_fmq = 0.0
    sum_hl_hiql = sum_hl_wgsp = 0.0
    sum_fmq_gnorm = sum_fmq_J = sum_fmq_adisp = 0.0
    sum_v_std_ep = sum_v_std_dat = 0.0
    sum_mu_norm = sum_rep_pair = sum_mu_tgt_cos = 0.0
    sum_z_k_disp = sum_closer = sum_z_k_std = 0.0
    sum_v_mean_range = sum_J_std_N = 0.0
    sum_w_hl_max = sum_w_hl_ent = 0.0
    audit_log_n = diag_log_n = fmq_log_n = log_n = 0
    prev_phase = 1

    for step in range(total_steps):
        phase = 1 if step < warmup_steps else 2
        if phase != prev_phase:
            print(f"\n>>> Phase 2 begins at step {step:,} — WGSP+FMQ active\n",
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
        ll_fmq = None

        if use_wgsp and phase == 2:
            z_t, z_target, g_ult = real_cache.sample_hl_batch(
                batch_size, subgoal_steps)

            hl_loss, rollouts = hl_wgsp_step(
                z_t=z_t, g_ult=g_ult,
                hl_actor=hl_actor, ll_actor=ll_actor, decoder=decoder,
                wm_model=wm_model, goal_rep=goal_rep, value_net=value_net,
                N=N, M=M, k=k_plan,
                beta_geom=beta_geom,
                alpha_H=alpha_H_wgsp, alpha_L=1.0,   # u_ll not used in FMQ
                action_scale=action_scale,
                use_decoder=use_decoder,
                use_geometric_term=use_geometric_term,
                use_v_in_J=use_v_in_J,
                lambda_anchor=lambda_anchor, z_target=z_target,
                return_rollouts=True,          # always needed for FMQ i* selection
                lambda_mopo=lambda_mopo,
                return_v_diag=True,
                ll_score_mode='goal',
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

            # ── FMQ LL step ──────────────────────────────────────────────────
            if use_fmq and rollouts is not None:
                rep_cand  = rollouts['rep_cand']    # [B, N, rep_dim]
                J_per_rep = rollouts['J_per_rep']   # [B, N]
                B = rollouts['B']

                i_star   = J_per_rep.argmax(dim=1)  # [B]
                rep_star = rep_cand[torch.arange(B, device=device), i_star]

                ll_fmq, fmq_diag = ll_fmq_step(
                    z_t=z_t, g_ult=g_ult, rep_star=rep_star,
                    ll_actor=ll_actor, decoder=decoder, wm_model=wm_model,
                    value_net=value_net, adapter=adapter,
                    action_scale=action_scale, eta_fmq=eta_fmq,
                    use_decoder=use_decoder, return_diag=True,
                )
                sum_ll_fmq += ll_fmq.item()
                if fmq_diag is not None:
                    sum_fmq_gnorm += fmq_diag['grad_norm']
                    sum_fmq_J     += fmq_diag['J_LL_mean']
                    sum_fmq_adisp += fmq_diag['a_displacement']
                    fmq_log_n += 1

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
        if ll_fmq is not None:
            total_loss = total_loss + lambda_fmq * ll_fmq

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
            fd  = max(fmq_log_n, 1)
            now = time.time()
            sps = log_interval / max(now - t_int, 1e-6)
            remaining = (total_steps - step) / max(sps, 1e-6)
            elapsed = now - t0

            print(
                f"Step {step:06d}/{total_steps:,} ({100.*step/total_steps:4.1f}%) "
                f"| Ph{phase} | "
                f"V {sum_v/(2*d):.4f} | "
                f"LL_h {sum_ll_hiql/d:.4f} LL_fmq {sum_ll_fmq/d:.4f} | "
                f"HL_h {sum_hl_hiql/d:.4f} HL_w {sum_hl_wgsp/d:.4f} | "
                f"FMQ g‖{sum_fmq_gnorm/fd:.3f} J {sum_fmq_J/fd:.2f} "
                f"aΔ {sum_fmq_adisp/fd:.3f} | "
                f"Vstd ep/dat {sum_v_std_ep/ad:.3f}/{sum_v_std_dat/ad:.3f} | "
                f"closer {sum_closer/dd:.3f} Jσ {sum_J_std_N/dd:.3f} | "
                f"wmax {sum_w_hl_max/dd:.3f} | "
                f"{sps:.1f} sps | ETA {remaining/60:.0f}m",
                flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, phase, sum_v/(2*d),
                    sum_ll_hiql/d, sum_ll_fmq/d,
                    sum_hl_hiql/d, sum_hl_wgsp/d,
                    sum_fmq_gnorm/fd, sum_fmq_J/fd, sum_fmq_adisp/fd,
                    now - t_int,
                    sum_v_std_ep/ad, sum_v_std_dat/ad,
                    sum_mu_norm/dd, sum_mu_tgt_cos/dd, sum_rep_pair/dd,
                    sum_z_k_std/dd, sum_z_k_disp/dd, sum_closer/dd,
                    sum_v_mean_range/dd, sum_J_std_N/dd,
                    sum_w_hl_max/dd, sum_w_hl_ent/dd,
                ])
            # Reset accumulators
            sum_v = sum_ll_hiql = sum_ll_fmq = 0.0
            sum_hl_hiql = sum_hl_wgsp = 0.0
            sum_fmq_gnorm = sum_fmq_J = sum_fmq_adisp = 0.0
            sum_v_std_ep = sum_v_std_dat = 0.0
            sum_mu_norm = sum_rep_pair = sum_mu_tgt_cos = 0.0
            sum_z_k_disp = sum_closer = sum_z_k_std = 0.0
            sum_v_mean_range = sum_J_std_N = 0.0
            sum_w_hl_max = sum_w_hl_ent = 0.0
            audit_log_n = diag_log_n = fmq_log_n = log_n = 0
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

    parser = argparse.ArgumentParser(description='FMQ-WGSP training')

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
    parser.add_argument('--M',               type=int,   default=1,
                        help='Rollouts per candidate for HL scoring. '
                             'Default 1 (sufficient with soft AWR; '
                             'within-rep advantage not used in FMQ-WGSP).')
    parser.add_argument('--batch_size',      type=int,   default=256)

    # IQL / AWR temperatures
    parser.add_argument('--alpha_H',         type=float, default=3.0)
    parser.add_argument('--alpha_L',         type=float, default=3.0)
    parser.add_argument('--alpha_H_wgsp',    type=float, default=3.0)
    parser.add_argument('--beta_geom',       type=float, default=0.0)
    parser.add_argument('--lambda_anchor',   type=float, default=0.0)

    # FMQ-specific
    parser.add_argument('--eta_fmq',         type=float, default=0.1,
                        help='FMQ trust-region radius. '
                             'Sweep [0.02, 0.05, 0.1, 0.2].')
    parser.add_argument('--lambda_fmq',      type=float, default=1.0,
                        help='Weight on FMQ LL loss.')
    parser.add_argument('--lambda_hiql_ll',  type=float, default=1.0,
                        help='Weight on HIQL LL anchor loss.')

    # RL params
    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--expectile',     type=float, default=0.7)
    parser.add_argument('--action_scale',  type=float, default=3.0)
    parser.add_argument('--rep_dim',       type=int,   default=10)

    # Toggles
    parser.add_argument('--use_wgsp',           type=_bool, default=True)
    parser.add_argument('--use_fmq',            type=_bool, default=True)
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
        f'./checkpoints_fmq_wgsp_k{args.subgoal_steps}_N{args.N}_M{args.M}'
        f'_eta{args.eta_fmq}_lfmq{args.lambda_fmq}'
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

    fmq_train_loop(
        wm_model=wm_model,
        hl_actor=hl_actor, ll_actor=ll_actor,
        value_net=value_net, value_target=value_target,
        goal_rep=goal_rep, goal_rep_target=goal_rep_target,
        real_cache=real_cache, decoder=decoder,
        optimizer=optimizer, trainable_params=trainable_params,
        use_wgsp=args.use_wgsp, use_fmq=args.use_fmq,
        use_decoder=args.use_decoder,
        use_geometric_term=args.use_geometric_term,
        use_v_in_J=args.use_v_in_J,
        total_steps=args.total_steps,
        warmup_fraction=args.warmup_fraction,
        subgoal_steps=args.subgoal_steps, k_plan=args.k_plan,
        N=args.N, M=args.M, batch_size=args.batch_size,
        gamma=args.gamma, tau=0.005, expectile=args.expectile,
        alpha_H=args.alpha_H, alpha_L=args.alpha_L,
        alpha_H_wgsp=args.alpha_H_wgsp,
        beta_geom=args.beta_geom, lambda_anchor=args.lambda_anchor,
        action_scale=args.action_scale, rep_dim=REP_DIM,
        save_dir=save_dir, device=device,
        eta_fmq=args.eta_fmq, lambda_fmq=args.lambda_fmq,
        lambda_hiql_ll=args.lambda_hiql_ll,
        lambda_mopo=args.lambda_mopo,
        adapter=adapter,
        hl_wgsp_coeff=args.hl_wgsp_coeff,
        lambda_hiql_reg=args.lambda_hiql_reg,
    )
