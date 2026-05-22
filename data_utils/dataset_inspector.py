import os
import sys
import numpy as np
import stable_worldmodel as swm

def inspect_dataset():
    print("--- OGBench Dataset Inspector ---")
    
    # Grab the ephemeral path just like in your train.py
    stablewm_home = os.environ.get("STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))

    # Standard path for the dataset
    data_path = os.path.join(stablewm_home, "ogbench", "cube_single_expert")
    
    print(f"Loading dataset from: {data_path}.h5 ...\n")
    
    try:
        # Load the dataset using the SWM wrapper
        dataset = swm.data.HDF5Dataset(data_path)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # 1. Total number of episodes (videos)
    num_episodes = len(dataset.lengths)
    
    # 2. Episode lengths
    lengths = dataset.lengths
    min_len = np.min(lengths)
    max_len = np.max(lengths)
    avg_len = np.mean(lengths)
    
    # 3. Total transitions in the dataset
    total_transitions = np.sum(lengths)
    
    print("=== DATASET STATISTICS ===")
    print(f"Total Expert Videos (Episodes) : {num_episodes}")
    print(f"Total Transitions (Frames)     : {total_transitions:,}")
    print("-" * 26)
    print(f"Shortest Video Length          : {min_len} frames")
    print(f"Longest Video Length           : {max_len} frames")
    print(f"Average Video Length           : {avg_len:.2f} frames")
    print("==========================\n")
    
    # --- LATENT DISTANCE ANALYZER ---
    print("--- Analyzing Latent Space Distances ---")
    cache_path = os.path.join(ephemeral, "stable_wm_data", "cube_all_latents_cache.pt")
    
    if os.path.exists(cache_path):
        import torch
        print("Latent cache found! Computing threshold math...")
        cache = torch.load(cache_path, map_location="cpu")
        latents = cache['all_latents']
        
        gap1_dists = []
        gap10_dists = []
        max_dists = []
        
        for ep in latents: # Check first 1000 episodes to be fast
            if len(ep) < 15: continue
            
            # L2 distance between frame[t] and frame[t+1]
            gap1_dists.append(torch.norm(ep[1:] - ep[:-1], p=2, dim=-1).mean().item())
            # L2 distance between frame[t] and frame[t+10]
            gap10_dists.append(torch.norm(ep[10:] - ep[:-10], p=2, dim=-1).mean().item())
            # L2 distance from start to end
            max_dists.append(torch.norm(ep[-1] - ep[0], p=2, dim=-1).item())
            
            if len(gap1_dists) >= 1000:
                break
                
        print(f"Average 1-Step Distance  : {np.mean(gap1_dists):.4f}")
        print(f"Average 10-Step Distance : {np.mean(gap10_dists):.4f}")
        print(f"Average Max Distance     : {np.mean(max_dists):.4f}")
        print("\nThreshold Rule of Thumb: Your environment threshold (e.g. 0.5) should be")
        print("slightly larger than the 1-Step Distance, but much smaller than the Max Distance.")
        print("----------------------------------------\n")
    else:
        print("Latent cache not found. Run precompute_latents.py to analyze latent distances.\n")

    # Optional: Look at the actual keys stored in a single chunk
    print("Inspecting data structure of the first frame...")
    sample_chunk = dataset.load_chunk(np.array([0]), np.array([0]), np.array([1]))[0]
    
    print("Available Tensors per frame:")
    for key, value in sample_chunk.items():
        if isinstance(value, np.ndarray) or hasattr(value, 'shape'):
            print(f"  - {key}: shape {value.shape}")
        else:
            print(f"  - {key}: {type(value)}")

if __name__ == "__main__":
    # Ensure the parent directory is in the path if needed for swm imports
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    inspect_dataset()