"""
Goal-conditioned SAC trained with world-model rollouts, sparse reward + HER.

Identical algorithm to sac_env_train.py — same sparse reward, same HER relabeling,
same action_scale=1.0, same goal sampling (random future frame from episode).
The only difference is that z_next comes from the WM predictor instead of
encoding a real environment observation.

This is the right comparison: the two scripts isolate ONLY the rollout source
(WM prediction vs real env + JEPA encoder), keeping everything else identical.

Usage:
    python latent_hindsight_rl/sac_wm_train.py \\
        --ckpt_path   $STABLEWM_HOME/cube/lejepa \\
        --dataset_path $STABLEWM_HOME/ogbench/visual-cube-single-play-v0_224 \\
        --cache_path  $STABLEWM_HOME/ogbench/lewm_224_latents_cache.pt \\
        --img_size 224 --patch_size 14 \\
        --num_iters 5000 \\
        --save_dir ./checkpoints_wm_pure_distance

Evaluate with:
    python latent_hindsight_rl/eval_ogbench.py \\
        --img_size 224 --patch_size 14 \\
        --checkpoint_dir ./checkpoints_wm_pure_distance \\
        --dataset_path $STABLEWM_HOME/ogbench/visual-cube-single-play-v0_224
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import csv
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from sklearn import preprocessing

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


import stable_pretraining as spt
import stable_worldmodel as swm

from sac_baseline import GoalConditionedActor, TwinCritic, StandardReplayBuffer
# Reuse HER relabeling and unified JEPA loader from sac_env_train
from sac_env_train import her_relabel, load_jepa


# =============================================================================
# WM ROLLOUT  (mirror of collect_env_episode from sac_env_train.py)
# =============================================================================

def collect_wm_episode(jepa_model, all_latents, actor, T_max, device):
    """
    Roll out one episode using the world-model predictor.

    All transitions carry r=0, done=False — original-task reward is omitted
    since we don't trust latent L2 as a reliable task-success metric.
    HER relabeling (future strategy) provides all positive learning signal:
    a transition gets r=1 when ||z_next - z_her_goal|| < her_threshold.

    Returns:
        episode_latents : list of [192] tensors, length = T_max + 1
        transitions     : list of (z_t, action[25], z_next, r=0.0, done=0.0)
        z_goal          : [192] the episode's conditioning goal latent
    """
    ep_idx   = random.randint(0, len(all_latents) - 1)
    ep       = all_latents[ep_idx]
    ep_len   = ep.shape[0]

    if ep_len < 2:
        start_t = 0
        goal_t  = 0
    else:
        start_t = random.randint(0, ep_len - 2)
        goal_t  = random.randint(start_t + 1, ep_len - 1)

    z_curr  = ep[start_t].to(device)
    z_goal  = ep[goal_t].to(device)
    z_state = z_curr.unsqueeze(0).unsqueeze(0)   # [1, 1, 192]

    episode_latents = [z_curr.clone()]
    transitions     = []

    with torch.no_grad():
        for _ in range(T_max):
            action, _, _ = actor.sample(z_curr.unsqueeze(0), z_goal.unsqueeze(0))
            action = action.squeeze(0)

            act_in       = action.unsqueeze(0).unsqueeze(0)
            act_emb      = jepa_model.action_encoder(act_in)
            z_next_state = jepa_model.predict(z_state, act_emb)[:, -1:]
            z_next       = z_next_state.squeeze(0).squeeze(0)

            z_state = z_next_state
            episode_latents.append(z_next.clone())
            transitions.append((z_curr, action, z_next, 0.0, 0.0))
            z_curr = z_next

    return episode_latents, transitions, z_goal


# =============================================================================
# TRAINING LOOP  (identical SAC update to sac_env_train.py)
# =============================================================================

def train_loop_wm(jepa_model, all_latents, actor, critic, critic_target,
                  actor_optimizer, critic_optimizer, alpha_optimizer,
                  log_alpha, target_entropy, replay_buffer,
                  num_iterations=5000, gamma=0.99, tau=0.005,
                  T_max=40, episodes_per_iter=5, her_k=4, her_threshold=2.0,
                  save_dir="./checkpoints_wm_pure_distance"):

    os.makedirs(save_dir, exist_ok=True)
    device = next(actor.parameters()).device

    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["Iteration", "Success_Rate", "Buffer_Size",
                                 "Actor_Loss", "Critic_Loss", "Total_Time"])

    recent_successes = deque(maxlen=episodes_per_iter * 20)
    start_time = time.time()

    for iteration in range(num_iterations):

        # --- ROLLOUT: collect episodes_per_iter WM episodes ---
        iter_transitions = []

        for _ in range(episodes_per_iter):
            ep_latents, ep_transitions, z_goal = collect_wm_episode(
                jepa_model, all_latents, actor, T_max=T_max, device=device,
            )
            # Success is measured at eval time via the real env, not during WM rollouts
            recent_successes.append(0.0)

            # Original transitions: r=0, done=False (HER provides all learning signal)
            for z_t, action, z_next, r_sparse, done in ep_transitions:
                iter_transitions.append(
                    (z_t, action, z_next, z_goal.clone(), r_sparse, done)
                )

            # HER relabeled transitions (compute_reward with latent L2 threshold)
            for z_t, action, z_next, z_her_goal, r_her, done_her in \
                    her_relabel(ep_latents, ep_transitions, K=her_k,
                                goal_threshold=her_threshold):
                iter_transitions.append(
                    (z_t, action, z_next, z_her_goal, r_her, done_her)
                )

        # Store batch to replay buffer
        if iter_transitions:
            z_currs  = torch.stack([t[0] for t in iter_transitions]).to(device)
            actions  = torch.stack([t[1] for t in iter_transitions]).to(device)
            z_nexts  = torch.stack([t[2] for t in iter_transitions]).to(device)
            z_goals  = torch.stack([t[3] for t in iter_transitions]).to(device)
            rewards  = torch.tensor([t[4] for t in iter_transitions],
                                    dtype=torch.float32, device=device)
            dones    = torch.tensor([t[5] for t in iter_transitions],
                                    dtype=torch.float32, device=device)
            replay_buffer.store_transitions(
                z_currs, actions, z_nexts, z_goals, rewards, dones
            )

        # --- SAC TRAINING (identical to sac_env_train.py) ---
        avg_actor_loss = avg_critic_loss = 0.0

        if replay_buffer.size >= 256:
            for _ in range(40):
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(256)
                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)
                alpha = log_alpha.exp().item()

                with torch.no_grad():
                    next_a, next_log_pi, _ = actor.sample(z_next_b, g_b)
                    tq1, tq2 = critic_target(z_next_b, g_b, next_a)
                    target_q  = r_b + (1.0 - d_b) * gamma * (
                        torch.min(tq1, tq2) - alpha * next_log_pi
                    )
                    # Clamp to valid range for sparse reward in [0,1]: Q ∈ [0, 1/(1-γ)]
                    target_q = target_q.clamp(0.0, 1.0 / (1.0 - gamma))

                q1, q2 = critic(z_b, g_b, a_b)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()
                avg_critic_loss += critic_loss.item()

                new_a, log_pi, _ = actor.sample(z_b, g_b)
                for p in critic.parameters():
                    p.requires_grad = False
                q1_new, q2_new = critic(z_b, g_b, new_a)
                actor_loss = (alpha * log_pi - torch.min(q1_new, q2_new)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
                avg_actor_loss += actor_loss.item()
                for p in critic.parameters():
                    p.requires_grad = True

                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                for tp, p in zip(critic_target.parameters(), critic.parameters()):
                    tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

        current_sr = float(np.mean(recent_successes)) if recent_successes else 0.0

        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            a_val = avg_actor_loss / 40.0 if replay_buffer.size >= 256 else 0.0
            c_val = avg_critic_loss / 40.0 if replay_buffer.size >= 256 else 0.0
            print(f"Iter {iteration:05d} | SR: {current_sr*100:.1f}% | "
                  f"Act: {a_val:.4f} | Crit: {c_val:.4f} | "
                  f"Buf: {replay_buffer.size} | t: {elapsed:.1f}s")
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([iteration, current_sr, replay_buffer.size,
                                         a_val, c_val, elapsed])
            start_time = time.time()

        if (iteration > 0 and iteration % 100 == 0) or iteration == num_iterations - 1:
            torch.save(actor.state_dict(),  os.path.join(save_dir, "actor_policy.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, "critic_network.pth"))
            print(f"  --> Checkpoint saved at iteration {iteration}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Goal-conditioned SAC with WM rollouts, sparse reward + HER"
    )
    parser.add_argument("--ckpt_path",       required=True,
                        help="Path to lewm_ogbench_weights.ckpt")
    parser.add_argument("--dataset_path",    required=True,
                        help="Path to OGBench HDF5 dataset (for action scaler fitting)")
    parser.add_argument("--cache_path",      default=None,
                        help="Path to pre-computed latent cache .pt file. "
                             "Default: ~/stable_wm_data/ogbench/lewm_224_latents_cache.pt")
    parser.add_argument("--save_dir",        default="./checkpoints_wm_pure_distance")
    parser.add_argument("--num_iters",       type=int, default=5000)
    parser.add_argument("--T_max",           type=int, default=40)
    parser.add_argument("--her_k",           type=int,   default=4)
    parser.add_argument("--her_threshold",   type=float, default=2.0,
                        help="Latent L2 threshold for HER compute_reward (same as env version)")
    parser.add_argument("--episodes_per_iter", type=int, default=5,
                        help="WM episodes per training iteration (matches sac_env_train.py's "
                             "5 task IDs per iteration)")
    parser.add_argument("--img_size",        type=int, default=224,
                        help="224 → AutoCostModel; 64 → .ckpt state-dict loader")
    parser.add_argument("--patch_size",      type=int, default=14,
                        help="ViT patch size (14 for 224x224, 8 for 64x64)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- JEPA model (for WM predictor) ---
    print("Loading JEPA / WM...")
    jepa_model = load_jepa(args.ckpt_path, device=device,
                           img_size=args.img_size, patch_size=args.patch_size)

    # --- Pre-computed latent cache ---
    if args.cache_path is None:
        args.cache_path = os.path.join(
            os.path.expanduser("~"), "stable_wm_data",
            "ogbench", "lewm_224_latents_cache.pt"
        )
    print(f"Loading latent cache from {args.cache_path}...")
    cache = torch.load(args.cache_path, map_location="cpu")
    all_latents = cache["all_latents"]
    print(f"  Loaded {len(all_latents)} episodes.")

    # --- Action scaler (for eval compatibility — not used during WM training) ---
    print(f"Loading dataset from {args.dataset_path} to fit action scaler...")
    dataset     = swm.data.HDF5Dataset(args.dataset_path)
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(action_data)
    print(f"  Action scaler fitted on {len(action_data)} rows.")

    # --- Save training config (action_scale=1.0 for eval_ogbench.py compatibility) ---
    os.makedirs(args.save_dir, exist_ok=True)
    training_config = {"action_scale": 1.0, "latent_dim": 192, "action_dim": 25,
                       "reward_type": "sparse_her_wm"}
    with open(os.path.join(args.save_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Saved training_config.json to {args.save_dir}")

    # --- SAC components (action_scale=1.0, matching sac_env_train.py) ---
    actor         = GoalConditionedActor(action_dim=25, action_scale=1.0).to(device)
    critic        = TwinCritic(action_dim=25).to(device)
    critic_target = TwinCritic(action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy   = -float(actor.mean_linear.out_features)  # -25
    log_alpha        = torch.tensor([-2.0], requires_grad=True, device=device)
    alpha_optimizer  = torch.optim.Adam([log_alpha], lr=3e-4)

    replay_buffer = StandardReplayBuffer(
        latent_dim=192, action_dim=25, capacity=1_000_000, device=device
    )

    print(f"\nStarting WM training | iters={args.num_iters} | T_max={args.T_max} | "
          f"HER_K={args.her_k} | HER_threshold={args.her_threshold} | "
          f"save_dir={args.save_dir}\n")

    train_loop_wm(
        jepa_model=jepa_model,
        all_latents=all_latents,
        actor=actor, critic=critic, critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        log_alpha=log_alpha,
        target_entropy=target_entropy,
        replay_buffer=replay_buffer,
        num_iterations=args.num_iters,
        T_max=args.T_max,
        episodes_per_iter=args.episodes_per_iter,
        her_k=args.her_k,
        her_threshold=args.her_threshold,
        save_dir=args.save_dir,
    )

    print("\nTraining complete.")
