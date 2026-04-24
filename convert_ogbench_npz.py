"""
Convert OGBench cube-single-play NPZ dataset to the HDF5 format expected by
stable_worldmodel's HDF5Dataset.

NPZ layout (flat, 1001 steps × 1000 episodes = 1,001,000 rows):
  observations  (1001000, 64, 64, 3)  uint8   HWC images
  actions       (1001000, 5)          float32
  terminals     (1001000,)            bool    True at last step of each episode
  qpos          (1001000, 21)         float32
  qvel          (1001000, 20)         float32

HDF5 layout required by HDF5Dataset:
  ep_len   (1000,)      int64   length of each episode
  ep_offset(1000,)      int64   cumulative start index of each episode
  pixels   (1001000, 64, 64, 3) uint8   HWC images (renamed from observations)
  action   (1001000, 5) float32         (renamed from actions)
  qpos     (1001000, 21) float32
  qvel     (1001000, 20) float32

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

    observations = data["observations"]   # (N, 64, 64, 3) uint8
    actions      = data["actions"]        # (N, 5)         float32
    terminals    = data["terminals"]      # (N,)            bool
    qpos         = data["qpos"]           # (N, 21)         float32
    qvel         = data["qvel"]           # (N, 20)         float32
    N = observations.shape[0]
    print(f"  Loaded {N:,} frames in {time.time()-t0:.1f}s")

    # --- Episode boundaries from terminals ---
    term_indices = np.where(terminals)[0]              # last step of each episode
    ep_starts    = np.concatenate([[0], term_indices + 1])[:-1]  # start of each episode
    ep_lens      = np.diff(np.concatenate([ep_starts, [N]])).astype(np.int64)
    ep_offset    = ep_starts.astype(np.int64)
    num_eps      = len(ep_lens)

    print(f"  Episodes: {num_eps}, lengths {ep_lens.min()}–{ep_lens.max()}")
    assert ep_lens.min() == ep_lens.max() == 1001, \
        f"Expected uniform 1001-step episodes, got {ep_lens.min()}–{ep_lens.max()}"

    # --- Write HDF5 ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nWriting HDF5 to: {output_path}")

    with h5py.File(output_path, "w") as f:
        # Metadata
        f.create_dataset("ep_len",    data=ep_lens,   dtype=np.int64)
        f.create_dataset("ep_offset", data=ep_offset, dtype=np.int64)

        # Data arrays — write in chunks to avoid peak RAM = 2× dataset
        img_ds    = f.create_dataset("pixels", shape=(N, 64, 64, 3),
                                     dtype=np.uint8,   chunks=(256, 64, 64, 3),
                                     compression="gzip", compression_opts=1)
        action_ds = f.create_dataset("action", shape=(N, 5),
                                     dtype=np.float32, chunks=(4096, 5))
        qpos_ds   = f.create_dataset("qpos",   shape=(N, 21),
                                     dtype=np.float32, chunks=(4096, 21))
        qvel_ds   = f.create_dataset("qvel",   shape=(N, 20),
                                     dtype=np.float32, chunks=(4096, 20))

        ep_chunk = chunk_episodes
        steps_per_chunk = ep_chunk * 1001

        for start_ep in range(0, num_eps, ep_chunk):
            end_ep    = min(start_ep + ep_chunk, num_eps)
            row_start = int(ep_offset[start_ep])
            row_end   = row_start + int((end_ep - start_ep) * 1001)

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
        print(f"  qpos:      {f['qpos'].shape}")
        print(f"  qvel:      {f['qvel'].shape}")
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
