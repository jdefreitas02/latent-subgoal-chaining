"""
Convert OGBench NPZ dataset to the HDF5 format expected by stable_worldmodel's HDF5Dataset.

Works for any OGBench environment (cube-single, cube-double, etc.) — dimensions
are inferred from the NPZ file automatically.

NPZ layout (flat, ep_len steps × num_eps episodes rows):
  observations  (N, H, W, 3)     uint8   HWC images
  actions       (N, action_dim)  float32
  terminals     (N,)             bool    True at last step of each episode
  qpos          (N, qpos_dim)    float32
  qvel          (N, qvel_dim)    float32

HDF5 layout required by HDF5Dataset:
  ep_len    (num_eps,)     int64   length of each episode
  ep_offset (num_eps,)     int64   cumulative start index of each episode
  pixels    (N, H, W, 3)  uint8   HWC images (renamed from observations)
  action    (N, action_dim) float32 (renamed from actions)
  qpos      (N, qpos_dim) float32
  qvel      (N, qvel_dim) float32

Usage:
    python convert_ogbench_npz.py [--input PATH] [--output PATH]

Defaults (use $STABLEWM_HOME env var, falls back to $HOME/stable_wm_data):
    input:  $STABLEWM_HOME/ogbench/visual-cube-single-play-v0.npz
    output: $STABLEWM_HOME/ogbench/cube_single_play_v0.h5
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np


def convert(input_path: str, output_path: str, chunk_episodes: int = 50) -> None:
    print(f"Loading NPZ from: {input_path}")
    t0 = time.time()
    data = np.load(input_path)

    observations = data["observations"]   # (N, H, W, 3) uint8
    actions      = data["actions"]        # (N, action_dim) float32
    terminals    = data["terminals"]      # (N,)            bool
    qpos         = data["qpos"]           # (N, qpos_dim)   float32
    qvel         = data["qvel"]           # (N, qvel_dim)   float32
    N = observations.shape[0]
    H, W        = observations.shape[1], observations.shape[2]
    action_dim  = actions.shape[1]
    qpos_dim    = qpos.shape[1]
    qvel_dim    = qvel.shape[1]
    print(f"  Loaded {N:,} frames in {time.time()-t0:.1f}s")
    print(f"  Image: {H}×{W}  action_dim={action_dim}  qpos_dim={qpos_dim}  qvel_dim={qvel_dim}")

    # --- Episode boundaries from terminals ---
    term_indices = np.where(terminals)[0]              # last step of each episode
    ep_starts    = np.concatenate([[0], term_indices + 1])[:-1]  # start of each episode
    ep_lens      = np.diff(np.concatenate([ep_starts, [N]])).astype(np.int64)
    ep_offset    = ep_starts.astype(np.int64)
    num_eps      = len(ep_lens)

    ep_len_min, ep_len_max = int(ep_lens.min()), int(ep_lens.max())
    print(f"  Episodes: {num_eps}, lengths {ep_len_min}–{ep_len_max}")
    if ep_len_min != ep_len_max:
        print(f"  WARNING: non-uniform episode lengths ({ep_len_min}–{ep_len_max})")
    else:
        print(f"  All episodes: {ep_len_min} steps (uniform)")
    # --- Write HDF5 ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nWriting HDF5 to: {output_path}")

    with h5py.File(output_path, "w") as f:
        # Metadata
        f.create_dataset("ep_len",    data=ep_lens,   dtype=np.int64)
        f.create_dataset("ep_offset", data=ep_offset, dtype=np.int64)

        # Data arrays — write in chunks to avoid peak RAM = 2× dataset
        img_ds    = f.create_dataset("pixels", shape=(N, H, W, 3),
                                     dtype=np.uint8,   chunks=(256, H, W, 3),
                                     compression="gzip", compression_opts=1)
        action_ds = f.create_dataset("action", shape=(N, action_dim),
                                     dtype=np.float32, chunks=(4096, action_dim))
        qpos_ds   = f.create_dataset("qpos",   shape=(N, qpos_dim),
                                     dtype=np.float32, chunks=(4096, qpos_dim))
        qvel_ds   = f.create_dataset("qvel",   shape=(N, qvel_dim),
                                     dtype=np.float32, chunks=(4096, qvel_dim))

        ep_chunk = chunk_episodes

        for start_ep in range(0, num_eps, ep_chunk):
            end_ep    = min(start_ep + ep_chunk, num_eps)
            row_start = int(ep_offset[start_ep])
            # Use actual offsets/lengths rather than assuming uniform ep length
            row_end   = (int(ep_offset[end_ep]) if end_ep < num_eps
                         else N)

            img_ds   [row_start:row_end] = observations[row_start:row_end]
            action_ds[row_start:row_end] = actions     [row_start:row_end]
            qpos_ds  [row_start:row_end] = qpos        [row_start:row_end]
            qvel_ds  [row_start:row_end] = qvel        [row_start:row_end]

            pct = end_ep / num_eps * 100
            print(f"  {end_ep}/{num_eps} episodes ({pct:.0f}%)", end="\r", flush=True)

    size_gb = os.path.getsize(output_path) / 1e9
    print(f"\nDone. HDF5 size: {size_gb:.2f} GB  ({time.time()-t0:.1f}s total)")

    # --- Quick verification ---
    print("\nVerifying HDF5...")
    with h5py.File(output_path, "r") as f:
        print(f"  Keys:      {list(f.keys())}")
        print(f"  ep_len:    {f['ep_len'].shape}  first 3: {f['ep_len'][:3]}")
        print(f"  ep_offset: {f['ep_offset'].shape}  first 3: {f['ep_offset'][:3]}")
        print(f"  pixels:    {f['pixels'].shape}  dtype={f['pixels'].dtype}")
        print(f"  action:    {f['action'].shape}  dtype={f['action'].dtype}")
        print(f"  qpos:      {f['qpos'].shape}  (qpos_dim={qpos_dim})")
        print(f"  qvel:      {f['qvel'].shape}  (qvel_dim={qvel_dim})")
        # Sample a random frame to confirm round-trip integrity
        idx = 12345
        orig = observations[idx]
        conv = f["pixels"][idx]
        assert np.array_equal(orig, conv), "Pixel round-trip mismatch!"
        print(f"  Pixel round-trip check at row {idx}: OK")
    print("Verification passed.")


def main():
    stablewm_home  = os.environ.get("STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    default_input  = os.path.join(stablewm_home, "ogbench", "visual-cube-single-play-v0.npz")
    default_output = os.path.join(stablewm_home, "ogbench", "cube_single_play_v0.h5")

    parser = argparse.ArgumentParser(description="Convert OGBench NPZ to HDF5 for LeWM training")
    parser.add_argument("--input",  default=default_input,  help="Path to input NPZ file")
    parser.add_argument("--output", default=default_output, help="Path for output HDF5 file")
    parser.add_argument("--chunk_episodes", type=int, default=50,
                        help="Episodes to process per write batch (tune for RAM)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output, chunk_episodes=args.chunk_episodes)


if __name__ == "__main__":
    main()
