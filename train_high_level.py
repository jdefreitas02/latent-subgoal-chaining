import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ── Sinusoidal time embedding (for diffusion) ─────────────────────────────────
def sinusoidal_embedding(t, dim):
    """Standard sinusoidal positional encoding, maps integer timesteps → [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]


# ── Dataset ───────────────────────────────────────────────────────────────────
class WaypointDataset(Dataset):
    """
    Extracts (z_start, z_subgoal, z_final) triplets from expert latent trajectories.

    For every episode and every valid start index t:
        z_start   = ep[t]          ← current state (input condition)
        z_subgoal = ep[t + gap]    ← what the high-level should predict
        z_final   = ep[-1]         ← ultimate goal (input condition)

    Stores only integer index pairs (ep_idx, t) to avoid duplicating the latent
    cache in memory — the actual tensors are looked up on __getitem__.
    """
    def __init__(self, all_latents, gap):
        self.all_latents = all_latents
        self.gap = gap
        self.indices = []
        for ep_idx, ep in enumerate(all_latents):
            T = ep.shape[0]
            if T <= gap:
                continue
            for t in range(T - gap):
                self.indices.append((ep_idx, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        ep = self.all_latents[ep_idx]
        return ep[t].float(), ep[t + self.gap].float(), ep[-1].float()


# ── MLP High-Level Policy ─────────────────────────────────────────────────────
class MLPHighLevel(nn.Module):
    """
    Directly regresses z_subgoal from (z_curr, z_goal) via a two-hidden-layer MLP.
    Trained with MSE loss against expert waypoints.
    """
    def __init__(self, latent_dim=192, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),          nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_curr, z_goal):
        return self.net(torch.cat([z_curr, z_goal], dim=-1))

    @torch.no_grad()
    def predict(self, z_curr, z_goal):
        return self.forward(z_curr, z_goal)


# ── Diffusion High-Level Policy ───────────────────────────────────────────────
class DiffusionHighLevel(nn.Module):
    """
    Conditional DDPM that generates z_subgoal by iterative denoising,
    conditioned on (z_curr, z_goal).

    Handles multimodal subgoal distributions — e.g. multiple valid intermediate
    arm configurations — that an MLP would average into an off-manifold blur.

    Training : denoising score matching (predict noise ε).
    Inference : deterministic DDIM with `ddim_steps` steps (~10 is sufficient).
    """
    def __init__(self, latent_dim=192, hidden=512, T=200, time_emb_dim=128):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        # Input to noise predictor:
        #   noisy z_subgoal  : latent_dim
        #   time embedding   : time_emb_dim
        #   condition (cat)  : latent_dim * 2
        in_dim = latent_dim + time_emb_dim + (latent_dim * 2)

        self.noise_pred = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

        # Noise schedule (registered as buffers so they follow .to(device))
        betas     = torch.linspace(1e-4, 0.02, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas',     betas)
        self.register_buffer('alphas',    alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    def _noise_pred_forward(self, z_noisy, t, z_curr, z_goal):
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        cond  = torch.cat([z_curr, z_goal], dim=-1)
        x     = torch.cat([z_noisy, t_emb, cond], dim=-1)
        return self.noise_pred(x)

    def training_loss(self, z_subgoal, z_curr, z_goal):
        """Standard DDPM denoising objective."""
        B   = z_subgoal.shape[0]
        t   = torch.randint(0, self.T, (B,), device=z_subgoal.device)
        eps = torch.randn_like(z_subgoal)
        ab  = self.alpha_bar[t].unsqueeze(-1)
        z_noisy  = ab.sqrt() * z_subgoal + (1.0 - ab).sqrt() * eps
        eps_pred = self._noise_pred_forward(z_noisy, t, z_curr, z_goal)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def predict(self, z_curr, z_goal, ddim_steps=10):
        """Deterministic DDIM sampling — fast inference with few steps."""
        B    = z_curr.shape[0]
        z    = torch.randn(B, self.latent_dim, device=z_curr.device)
        step_ids = torch.linspace(self.T - 1, 0, ddim_steps).long()

        for i, t_val in enumerate(step_ids):
            t = torch.full((B,), t_val.item(), device=z.device, dtype=torch.long)
            eps_pred = self._noise_pred_forward(z, t, z_curr, z_goal)

            ab      = self.alpha_bar[t_val]
            ab_prev = (self.alpha_bar[step_ids[i + 1]]
                       if i + 1 < ddim_steps
                       else torch.tensor(1.0, device=z.device))

            # DDIM deterministic update
            z0_pred = (z - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-8)
            z = ab_prev.sqrt() * z0_pred + (1.0 - ab_prev).sqrt() * eps_pred

        return z


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, model_type, train_loader, val_loader, optimizer, device,
          num_epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for z_start, z_subgoal, z_final in train_loader:
            z_start   = z_start.to(device)
            z_subgoal = z_subgoal.to(device)
            z_final   = z_final.to(device)

            optimizer.zero_grad()

            if model_type == 'diffusion':
                loss = model.training_loss(z_subgoal, z_start, z_final)
            else:
                loss = F.mse_loss(model(z_start, z_final), z_subgoal)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        # For diffusion: val loss is the denoising objective (same scale as train loss).
        # To compare subgoal MSE across model types, see the eval script.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_start, z_subgoal, z_final in val_loader:
                z_start   = z_start.to(device)
                z_subgoal = z_subgoal.to(device)
                z_final   = z_final.to(device)

                if model_type == 'diffusion':
                    loss = model.training_loss(z_subgoal, z_start, z_final)
                else:
                    loss = F.mse_loss(model(z_start, z_final), z_subgoal)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch:03d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print(f"\nDone. Best val loss: {best_val_loss:.4f} | Saved to: {save_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a high-level subgoal policy from expert data.")
    parser.add_argument('--model_type', type=str, required=True, choices=['mlp', 'diffusion'],
                        help="Architecture to train.")
    parser.add_argument('--gap', type=int, required=True,
                        help="WM-step gap. Should match the low-level policy this will be paired with.")
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1,
                        help="Fraction of triplets held out for validation.")
    parser.add_argument('--cache_path', type=str, default=None,
                        help="Path to latents cache .pt. Default: {EPHEMERAL}/stable_wm_data/cube_all_latents_cache.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model_type} | Gap: {args.gap}")

    ephemeral = os.environ.get("EPHEMERAL")
    if ephemeral is None and args.cache_path is None:
        raise ValueError("Either --cache_path or the EPHEMERAL environment variable must be set")

    cache_path = args.cache_path or os.path.join(
        ephemeral, "stable_wm_data", "cube_all_latents_cache.pt")
    print(f"Loading latents from {cache_path}...")
    cache      = torch.load(cache_path, map_location='cpu')
    all_latents = cache['all_latents']

    dataset = WaypointDataset(all_latents, gap=args.gap)
    print(f"Dataset: {len(dataset):,} triplets extracted at gap={args.gap}")

    val_size   = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    if args.model_type == 'mlp':
        model = MLPHighLevel().to(device)
    else:
        model = DiffusionHighLevel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_dir  = f"./checkpoints_high_level/{args.model_type}_gap{args.gap}"

    train(model, args.model_type, train_loader, val_loader, optimizer, device,
          args.num_epochs, save_dir)
