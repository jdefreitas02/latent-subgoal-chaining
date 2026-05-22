"""
Goal-conditioned SAC trained with real OGBench environment rollouts.

Key design choices vs. the world-model variant (sac_baseline.py):
  - Sparse reward from info['success'] instead of dense latent distance.
  - Hindsight Experience Replay (HER, future strategy) for sample efficiency.
  - Random future-frame goals: each episode the env supplies a task goal;
    HER then relabels transitions with random future achieved states.
  - action_scale=1.0 (tanh → [-1,1]) so inverse_transform maps cleanly
    to physical OGBench action range without saturation.
  - One episode per task_id (5 tasks) per iteration; sequential stepping.

Usage:
    python latent_hindsight_rl/sac_env_train.py \\
        --ckpt_path ./lewm_ogbench_weights.ckpt \\
        --dataset_path $STABLEWM_HOME/ogbench/cube_single_play_v0 \\
        --num_iters 5000 \\
        --save_dir ./checkpoints_env_pure_distance

Evaluate the trained checkpoint with:
    python latent_hindsight_rl/eval_actor.py \\
        --config-name cube_64 \\
        ++checkpoint_dir=./checkpoints_env_pure_distance \\
        ++policy=./lewm_ogbench_weights.ckpt
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
from torchvision.transforms import v2 as transforms
from sklearn import preprocessing

# jepa/module live in the parent directory
_parent_dir = os.path.abspath(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# SAC class definitions live in this directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import stable_pretraining as spt
import stable_worldmodel as swm

from sac_baseline import GoalConditionedActor, TwinCritic, StandardReplayBuffer


# =============================================================================
# JEPA LOADER
# =============================================================================
def load_jepa(ckpt_path, device="cuda", img_size=224, patch_size=14):
    """
    Unified JEPA loader matching eval_ogbench.py's load_jepa() signature.

    img_size=224 → AutoCostModel (base path, no extension needed).
    img_size=64  → construct ViT-tiny + ARPredictor from Lightning .ckpt.
    """
    if img_size == 224:
        model = swm.policy.AutoCostModel(ckpt_path)
        print(f"  Loaded 224x224 JEPA via AutoCostModel from {ckpt_path}")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        epoch = ckpt.get("epoch", "?")
    else:
        raw_sd = dict(ckpt)
        epoch = "?"
    model.load_state_dict(raw_sd, strict=True)
    print(f"  Loaded 64x64 JEPA from {ckpt_path} (epoch {epoch})")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_img_transform():
    """ImageNet normalisation only — env renders natively at the correct resolution."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


# =============================================================================
# OBSERVATION ENCODING
# =============================================================================
@torch.no_grad()
def encode_obs(obs_hwc, model, transform, device):
    """
    Encode a single HWC uint8 observation → 192-D latent on device.
    obs_hwc: np.ndarray [H, W, C] uint8
    """
    t = transform(obs_hwc)                         # [C, H, W] float32
    t = t.unsqueeze(0).unsqueeze(0).to(device)     # [1, 1, C, H, W]
    z = model.encode({"pixels": t})["emb"][:, -1]  # [1, 192]
    return z.squeeze(0)                            # [192]


def _goal_from_info(info):
    """Extract goal pixel image from env info, trying common OGBench key names."""
    for key in ("goal", "target", "desired_goal"):
        if key in info and info[key] is not None:
            return info[key]
    raise KeyError(
        f"Could not find goal image in env info. Available keys: {list(info.keys())}"
    )


# =============================================================================
# HER RELABELING  (future strategy)
# =============================================================================
def her_relabel(episode_latents, episode_transitions, K=4, goal_threshold=2.0):
    """
    Apply HER 'future' strategy, mirroring SB3's HerReplayBuffer.

    SB3 calls env.compute_reward(next_achieved_goal, new_desired_goal, info) for
    every relabelled transition. For us that function is:
        r = 1 if ||z_next - z_her_goal|| < goal_threshold else 0
    This is a distance check, NOT a positional check — transitions where z_next
    happens to be near the HER goal get r=1 even if t_future > t+1.

    done follows the SB3 convention: it mirrors the stored episode done at that
    timestep (False for mid-episode transitions). Goal achievement is encoded in
    the reward, not the done flag, because OGBench doesn't terminate on success.

    episode_latents    : list of [192] tensors, length T+1  (z_0 … z_T)
    episode_transitions: list of (z_t, action, z_next, r_sparse, done) tuples
    K                  : HER relabelings per original transition
    goal_threshold     : L2 distance in latent space for compute_reward
    """
    T = len(episode_transitions)
    relabeled = []

    for t, (z_t, action, z_next, r_sparse, done) in enumerate(episode_transitions):
        for _ in range(K):
            t_future   = random.randint(t + 1, T)
            z_her_goal = episode_latents[t_future].clone()

            # compute_reward: r=1 if achieved is within threshold of desired goal
            dist  = torch.norm(z_next - z_her_goal, p=2).item()
            r_her = 1.0 if dist < goal_threshold else 0.0
            # done is from the episode (False for mid-episode); goal achievement
            # is in the reward, not the termination flag (OGBench doesn't
            # terminate on success, matching SB3's standard GoalEnv convention)
            done_her = done

            relabeled.append((z_t, action, z_next, z_her_goal, r_her, done_her))

    return relabeled


# =============================================================================
# EPISODE COLLECTION
# =============================================================================
def collect_env_episode(env, actor, jepa_model, transform, action_scaler,
                        task_id, T_max, device):
    """
    Run one episode in the OGBench environment.

    Each 'step' here corresponds to one actor call (25-D action) which expands
    to 5 physical env steps (action_block=5), matching the WM training setup.

    Returns:
        episode_latents  : list of [192] tensors, length = actual_steps + 1
        transitions      : list of (z_t, action[25], z_next, r_sparse, done)
        z_goal           : [192] encoded goal latent
        success          : bool — True if info['success'] was ever 1
    """
    obs, info = env.reset(options={"task_id": task_id})

    goal_img = _goal_from_info(info)
    z_goal   = encode_obs(goal_img, jepa_model, transform, device)
    z_curr   = encode_obs(obs, jepa_model, transform, device)

    episode_latents = [z_curr.clone()]
    transitions     = []
    success         = False

    for _ in range(T_max):
        with torch.no_grad():
            action, _, _ = actor.sample(z_curr.unsqueeze(0), z_goal.unsqueeze(0))
        action = action.squeeze(0)  # [25]

        # Convert 25-D actor output → 5 physical actions ∈ [-1, 1]
        # action_scale=1.0 so raw values are in [-1,1]; inverse_transform maps
        # to the dataset's physical distribution and we clip for safety.
        action_np = action.cpu().numpy().reshape(5, 5)           # [5 blocks, 5 joints]
        physical  = np.clip(action_scaler.inverse_transform(action_np), -1.0, 1.0)

        # Step the environment 5 times (one per joint-action block)
        terminated = truncated = False
        step_info  = info
        for phys_t in range(5):
            obs_new, _, terminated, truncated, step_info = env.step(physical[phys_t])
            if terminated or truncated:
                break

        z_next  = encode_obs(obs_new, jepa_model, transform, device)
        episode_latents.append(z_next.clone())

        r_sparse = float(step_info.get("success", 0.0))
        done     = terminated or truncated or bool(r_sparse)
        success  = success or bool(r_sparse)

        transitions.append((z_curr, action, z_next, r_sparse, float(done)))

        z_curr = z_next
        obs    = obs_new
        info   = step_info

        if done:
            break

    return episode_latents, transitions, z_goal, success


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_loop_env(env, actor, critic, critic_target,
                   actor_optimizer, critic_optimizer, alpha_optimizer,
                   log_alpha, target_entropy, replay_buffer,
                   jepa_model, transform, action_scaler,
                   num_iterations=5000, gamma=0.99, tau=0.005,
                   T_max=40, num_task_ids=5, her_k=4, her_threshold=2.0,
                   save_dir="./checkpoints_env_pure_distance"):

    os.makedirs(save_dir, exist_ok=True)
    device = next(actor.parameters()).device

    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["Iteration", "Success_Rate", "Buffer_Size",
                                 "Actor_Loss", "Critic_Loss", "Total_Time"])

    recent_successes = deque(maxlen=num_task_ids * 20)
    start_time = time.time()

    for iteration in range(num_iterations):

        # --- ROLLOUT: one episode per task ID ---
        iter_transitions = []

        for task_id in range(1, num_task_ids + 1):
            ep_latents, ep_transitions, z_goal, success = collect_env_episode(
                env, actor, jepa_model, transform, action_scaler,
                task_id=task_id, T_max=T_max, device=device,
            )
            recent_successes.append(float(success))

            # Original transitions with sparse env reward
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

        # --- SAC TRAINING (identical update to sac_baseline.py) ---
        avg_actor_loss = avg_critic_loss = 0.0

        if replay_buffer.size >= 256:
            for _ in range(40):
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(256)
                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)
                alpha = log_alpha.exp().item()

                # Critic update
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

                # Actor update
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

                # Alpha update
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                # Soft target update
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
        description="Goal-conditioned SAC with real OGBench env rollouts, sparse reward + HER"
    )
    parser.add_argument("--ckpt_path",    required=True,
                        help="Path to lewm_ogbench_weights.ckpt (JEPA encoder)")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to OGBench HDF5 dataset (for action scaler)")
    parser.add_argument("--save_dir",     default="./checkpoints_env_pure_distance",
                        help="Directory to save actor/critic checkpoints")
    parser.add_argument("--num_iters",    type=int, default=5000)
    parser.add_argument("--T_max",        type=int, default=40,
                        help="Max actor steps per episode (each = 5 physical env steps)")
    parser.add_argument("--her_k",         type=int,   default=4,
                        help="HER relabelings per transition")
    parser.add_argument("--her_threshold", type=float, default=2.0,
                        help="Latent L2 threshold for HER compute_reward — "
                             "r=1 iff ||z_next - z_her_goal|| < this value")
    parser.add_argument("--img_size",     type=int, default=224,
                        help="64 → visual-cube-single-v0; 224 → swm/OGBCube-v0")
    parser.add_argument("--patch_size",   type=int, default=14,
                        help="ViT patch size (8 for 64x64, 14 for 224x224)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- JEPA encoder ---
    print("Loading JEPA encoder...")
    jepa_model = load_jepa(args.ckpt_path, device=device,
                           img_size=args.img_size, patch_size=args.patch_size)
    transform  = make_img_transform()

    # --- Action scaler (fit on dataset actions, same as eval_ogbench.py) ---
    print(f"Loading dataset from {args.dataset_path} to fit action scaler...")
    dataset     = swm.data.HDF5Dataset(args.dataset_path,
                                       keys_to_cache=["action"],
                                       cache_dir=str(Path(args.dataset_path).parent))
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(action_data)
    print(f"  Action scaler fitted on {len(action_data)} rows.")

    # --- OGBench environment ---
    import gymnasium
    import ogbench  # noqa: F401 — registers OGBench envs
    if args.img_size == 64:
        env = gymnasium.make("visual-cube-single-v0")
        print("Created visual-cube-single-v0 environment (native 64x64).")
    else:
        env = gymnasium.make("swm/OGBCube-v0",
                             ob_type="pixels", env_type="single", visualize_info=False)
        print("Created swm/OGBCube-v0 environment (native 224x224).")

    # --- Save training config so eval_actor.py loads the correct action_scale ---
    os.makedirs(args.save_dir, exist_ok=True)
    training_config = {"action_scale": 1.0, "latent_dim": 192, "action_dim": 25,
                       "reward_type": "sparse_her"}
    with open(os.path.join(args.save_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Saved training_config.json to {args.save_dir}")

    # --- SAC components ---
    # action_scale=1.0: tanh output already in [-1,1], matching OGBench physical range
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

    print(f"\nStarting training | iters={args.num_iters} | T_max={args.T_max} | "
          f"HER_K={args.her_k} | HER_threshold={args.her_threshold} | "
          f"save_dir={args.save_dir}\n")

    train_loop_env(
        env=env,
        actor=actor, critic=critic, critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        log_alpha=log_alpha,
        target_entropy=target_entropy,
        replay_buffer=replay_buffer,
        jepa_model=jepa_model,
        transform=transform,
        action_scaler=action_scaler,
        num_iterations=args.num_iters,
        T_max=args.T_max,
        num_task_ids=5,
        her_k=args.her_k,
        her_threshold=args.her_threshold,
        save_dir=args.save_dir,
    )

    env.close()
    print("\nTraining complete.")
