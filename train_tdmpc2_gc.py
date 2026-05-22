"""
Offline goal-conditioned TD-MPC2 training on OGBench cube-single-play (64x64).

Trains both offline-safety variants:
  --offline_mode bc_reg : Variant A — TD-MPC2-native + BC anchor on policy
  --offline_mode iql    : Variant B — IQL expectile V + AWR policy

Saves:
  {save_dir}/config.pt   — pickled GCTDMPC2Config used for this run
  {save_dir}/model.pt    — model state dict (and step counter)
  {save_dir}/training_metrics.csv — per-step loss components

Usage:
  python latent_hindsight_rl/train_tdmpc2_gc.py \
      --offline_mode iql \
      --save_dir checkpoints_tdmpc2_gc_iql_s0 \
      --total_steps 200000 \
      --dataset_path $STABLEWM_HOME/ogbench/visual-cube-single-play-v0.h5
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from latent_hindsight_rl.tdmpc2.config import GCTDMPC2Config  # noqa
from latent_hindsight_rl.tdmpc2.gc_world_model import GCWorldModel  # noqa
from latent_hindsight_rl.tdmpc2.dataset import OGBenchOfflineDataset  # noqa
from latent_hindsight_rl.tdmpc2.state_dataset import OGBenchStateDataset  # noqa
from latent_hindsight_rl.tdmpc2 import losses as L  # noqa
from latent_hindsight_rl.tdmpc2.scale import RunningScale  # noqa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str, required=True,
                   help='Path to dataset (.h5 for rgb, .npz for state)')
    p.add_argument('--obs', choices=['rgb', 'state'], default='rgb',
                   help='Observation type: rgb (64x64 visual) or state (28-dim)')
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--total_steps', type=int, default=200_000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--offline_mode', choices=['bc_reg', 'iql'], default='iql')
    p.add_argument('--bc_reg_lambda', type=float, default=0.5,
                   help='BC anchor weight (Variant A only)')
    p.add_argument('--iql_expectile_tau', type=float, default=0.7,
                   help='IQL expectile (Variant B only)')
    p.add_argument('--iql_awr_alpha', type=float, default=3.0,
                   help='AWR temperature (Variant B only)')
    p.add_argument('--log_std_max', type=float, default=2.0,
                   help='Policy log-std upper bound. 0.0 caps action noise at std=1.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--save_every', type=int, default=10_000)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Config ----
    cfg = GCTDMPC2Config()
    cfg.offline_mode = args.offline_mode
    cfg.bc_reg_lambda = args.bc_reg_lambda
    cfg.iql_expectile_tau = args.iql_expectile_tau
    cfg.iql_awr_alpha = args.iql_awr_alpha
    cfg.log_std_max = args.log_std_max
    cfg.obs = args.obs
    if args.obs == 'state':
        cfg.obs_shape = {'state': (cfg.state_obs_dim,)}

    print(f'Offline mode: {cfg.offline_mode}  obs={cfg.obs}')
    print(f'  bc_reg_lambda={cfg.bc_reg_lambda} (Variant A only)')
    print(f'  iql_expectile_tau={cfg.iql_expectile_tau}, iql_awr_alpha={cfg.iql_awr_alpha} (Variant B only)')

    # ---- Dataset ----
    print(f'Loading dataset from {args.dataset_path}')
    if args.obs == 'state':
        ds = OGBenchStateDataset(
            npz_path=args.dataset_path,
            horizon=args.horizon,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        ds = OGBenchOfflineDataset(
            h5_path=args.dataset_path,
            horizon=args.horizon,
            batch_size=args.batch_size,
            device=device,
        )

    # ---- Model ----
    model = GCWorldModel(cfg).to(device)
    print(f'Model parameters: {model.total_params:,}')

    # ---- Optimizers ----
    # Encoder gets a smaller LR to keep representation stable; matches TD-MPC2.
    wm_param_groups = [
        {'params': model._encoder.parameters(), 'lr': args.lr * cfg.enc_lr_scale},
        {'params': model._dynamics.parameters()},
        {'params': model._reward.parameters()},
        {'params': model._Qs.parameters()},
        {'params': model._V.parameters()},  # only used by IQL but included for both
    ]
    wm_optim = torch.optim.Adam(wm_param_groups, lr=args.lr)
    pi_optim = torch.optim.Adam(model._pi.parameters(), lr=args.lr, eps=1e-5)

    # Running scale (Variant A only, for normalizing Q in policy loss)
    scale = RunningScale(cfg).to(device) if cfg.offline_mode == 'bc_reg' else None

    # ---- CSV log ----
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'training_metrics.csv')
    csv_columns = [
        'step', 'wm_loss', 'consistency_loss', 'reward_loss', 'q_value_loss',
        'v_expectile_loss', 'pi_loss', 'pi_main', 'pi_bc', 'pi_entropy',
        'pi_q_mean', 'awr_w_mean', 'awr_w_max', 'pi_log_p_mean',
        'wm_grad_norm', 'pi_grad_norm', 'sps',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_columns)

    # Save config (for reload at eval time)
    torch.save(cfg, os.path.join(args.save_dir, 'config.pt'))

    # ---- Training loop ----
    print(f'Starting training: {args.total_steps:,} steps')
    t_start = time.time()
    last_log_t = t_start

    for step in range(args.total_steps):
        batch = ds.sample_batch()

        # WM update
        wm_optim.zero_grad(set_to_none=True)
        wm_loss, info, zs = L.compute_world_model_losses(
            model, batch, cfg, gamma=cfg.gamma,
        )
        wm_loss.backward()
        wm_grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in wm_param_groups for p in g['params']],
            cfg.grad_clip_norm,
        )
        wm_optim.step()

        # Policy update (separate optimizer; uses detached zs)
        pi_optim.zero_grad(set_to_none=True)
        pi_loss, pi_info = L.compute_policy_loss(
            model, zs, batch, cfg, scale=scale,
        )
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            model._pi.parameters(), cfg.grad_clip_norm,
        )
        pi_optim.step()

        # Soft-update target nets
        model.soft_update_targets()

        # Periodic log
        if (step + 1) % args.log_every == 0 or step == 0:
            now = time.time()
            sps = args.log_every / max(now - last_log_t, 1e-6)
            last_log_t = now
            elapsed = now - t_start
            eta_min = (args.total_steps - step - 1) / max(sps, 1e-6) / 60

            row = [step + 1]
            row.append(info.get('wm_loss', torch.tensor(0.0)).item())
            row.append(info.get('consistency_loss', torch.tensor(0.0)).item())
            row.append(info.get('reward_loss', torch.tensor(0.0)).item())
            row.append(info.get('q_value_loss', torch.tensor(0.0)).item())
            row.append(info.get('v_expectile_loss', torch.tensor(0.0)).item())
            row.append(pi_info.get('pi_loss', torch.tensor(0.0)).item())
            row.append(pi_info.get('pi_main', torch.tensor(0.0)).item())
            row.append(pi_info.get('pi_bc', torch.tensor(0.0)).item())
            row.append(pi_info.get('pi_entropy', torch.tensor(0.0)).item())
            row.append(pi_info.get('pi_q_mean', torch.tensor(0.0)).item())
            row.append(pi_info.get('awr_w_mean', torch.tensor(0.0)).item())
            row.append(pi_info.get('awr_w_max', torch.tensor(0.0)).item())
            row.append(pi_info.get('pi_log_p_mean', torch.tensor(0.0)).item())
            row.append(float(wm_grad_norm))
            row.append(float(pi_grad_norm))
            row.append(sps)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

            print(
                f"Step {step+1:6d}/{args.total_steps:,} "
                f"({100*(step+1)/args.total_steps:.1f}%) | "
                f"wm={row[1]:.4f} cons={row[2]:.4f} r={row[3]:.4f} q={row[4]:.4f}"
                f"{f' v={row[5]:.4f}' if cfg.offline_mode == 'iql' else ''} | "
                f"pi={row[6]:.3f} ent={row[9]:.3f} | "
                f"wm_g={row[14]:.2f} pi_g={row[15]:.2f} | "
                f"{sps:.1f} sps | el={elapsed/60:.1f}m | ETA={eta_min:.0f}m",
                flush=True,
            )

        # Save checkpoint
        if (step + 1) % args.save_every == 0 or (step + 1) == args.total_steps:
            ckpt = {
                'model': model.state_dict(),
                'step': step + 1,
                'cfg': cfg,
            }
            torch.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
            print(f'  → checkpoint saved at step {step+1}')

    elapsed = time.time() - t_start
    print(f'Training done. Elapsed: {elapsed/60:.1f}m')


if __name__ == '__main__':
    main()
