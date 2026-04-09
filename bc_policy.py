import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
    """Deterministic goal-conditioned behavioural cloning policy.

    Outputs actions in the same tanh-scaled space as GoalConditionedActor
    (range: (-action_scale, action_scale)) so the BC regularisation loss in
    train.py is simply F.mse_loss(actor_action, bc_policy(z, g).detach()).
    """

    def __init__(self, latent_dim=192, action_dim=25, hidden_dim=256, action_scale=3.0):
        super().__init__()
        self.action_scale = action_scale

        self.linear1     = nn.Linear(latent_dim * 2, hidden_dim)
        self.ln1         = nn.LayerNorm(hidden_dim)
        self.linear2     = nn.Linear(hidden_dim, hidden_dim)
        self.ln2         = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
        # Small output-layer init — same rationale as GoalConditionedActor
        nn.init.uniform_(self.action_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.action_head.bias, 0)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        return torch.tanh(self.action_head(x)) * self.action_scale
