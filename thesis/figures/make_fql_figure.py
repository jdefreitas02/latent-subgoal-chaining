"""Generate the FQL pipeline background figure.

Single-panel overview of Flow Q-Learning (Park et al., 2025), adapted to
the notation of the thesis background section: BC flow u_xi integrated
over S Euler steps, one-step actor mu_psi trained by distillation plus
critic guidance, and TD-trained critic Q_theta.

Run:  python make_fql_figure.py
Outputs: fql_pipeline.pdf (for the thesis), fql_pipeline.png (preview).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 13,
    }
)

ARROW = dict(arrowstyle="-|>", color="#3a3a3a", lw=1.5,
             shrinkA=2, shrinkB=2, mutation_scale=14)
LOSS_ARROW = dict(arrowstyle="<|-|>", color="#c0392b", lw=1.6,
                  linestyle=(0, (4, 3)), shrinkA=2, shrinkB=2,
                  mutation_scale=13)
GUIDE_ARROW = dict(arrowstyle="-|>", color="#d97b29", lw=1.6,
                   linestyle=(0, (4, 3)), shrinkA=2, shrinkB=2,
                   mutation_scale=13, connectionstyle="arc3,rad=-0.25")

DATA_FC, DATA_EC = "#f2f2f2", "#8a8a8a"
FLOW_FC, FLOW_EC = "#dbe9f9", "#5b8bd0"
ACT_FC, ACT_EC = "#fde6c8", "#d99b46"
CRI_FC, CRI_EC = "#e8dff5", "#9b7fc7"


def box(ax, x, y, w, h, label, fc, ec, fs=12.5):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.4,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=fs)


def latent_node(ax, x, y, label, r=0.42):
    ax.add_patch(Circle((x, y), r, fc="white", ec="#3a3a3a", lw=1.4))
    ax.text(x, y, label, ha="center", va="center", fontsize=13)


def arrow(ax, x0, y0, x1, y1, props=ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=props)


fig, ax = plt.subplots(figsize=(9.8, 4.8))
ax.set_xlim(0, 11.6)
ax.set_ylim(0, 7.6)
ax.set_aspect("equal")
ax.axis("off")

# inputs shared by both policies
latent_node(ax, 0.95, 3.85, "$z$")
latent_node(ax, 0.95, 2.55, "$x_0$")
ax.text(0.95, 1.85, r"$x_0 \sim \mathcal{N}(0, I)$", color="#777777",
        ha="center", va="center", fontsize=9.5, style="italic")

# BC flow (top path)
box(ax, 3.75, 5.0, 2.6, 1.15, "BC flow  $u_\\xi$\n($S$ Euler steps)",
    FLOW_FC, FLOW_EC, fs=11.5)
box(ax, 6.85, 5.0, 1.8, 0.9, r"$\Phi_\xi(z, x_0)$", "white", "#3a3a3a")

# dataset actions and flow-matching loss
box(ax, 6.85, 7.0, 1.7, 0.9, r"$\mathbf{a} \sim \mathcal{D}$",
    DATA_FC, DATA_EC)
arrow(ax, 6.85, 6.5, 6.85, 5.5, LOSS_ARROW)
ax.text(7.3, 6.0, "$\\mathcal{L}_{\\mathrm{BC}}$", color="#c0392b",
        ha="left", va="center", fontsize=13)
ax.text(8.1, 6.0, "(flow matching)", color="#c0392b",
        ha="left", va="center", fontsize=10)

# one-step actor (bottom path)
box(ax, 3.75, 1.7, 2.6, 1.15, "one-step actor  $\\mu_\\psi$",
    ACT_FC, ACT_EC, fs=11.5)
box(ax, 6.85, 1.7, 1.8, 0.9, r"$\mu_\psi(z, x_0)$", "white", "#3a3a3a")

# input arrows (z and x0 fan out to both policies)
arrow(ax, 1.32, 4.05, 2.4, 4.75)
arrow(ax, 1.34, 2.75, 2.4, 4.45)
arrow(ax, 1.32, 3.62, 2.4, 1.95)
arrow(ax, 1.34, 2.38, 2.4, 1.62)

# policy outputs
arrow(ax, 5.1, 5.0, 5.9, 5.0)
arrow(ax, 5.1, 1.7, 5.9, 1.7)

# distillation loss between the two action outputs
arrow(ax, 6.85, 4.5, 6.85, 2.2, LOSS_ARROW)
ax.text(7.3, 3.55, "$\\mathcal{L}_{\\mathrm{distil}}$", color="#c0392b",
        ha="left", va="center", fontsize=13)
ax.text(7.3, 2.95, "(weight $\\alpha$)", color="#c0392b",
        ha="left", va="center", fontsize=10)

# critic
box(ax, 10.1, 1.7, 1.85, 0.95, "critic  $Q_\\theta$", CRI_FC, CRI_EC)
arrow(ax, 7.8, 1.7, 9.12, 1.7)
ax.annotate("", xy=(7.55, 1.18), xytext=(9.7, 1.12),
            arrowprops=GUIDE_ARROW)
ax.text(8.6, 0.35, "maximise $Q_\\theta(z,\\cdot)$", color="#d97b29",
        ha="center", va="center", fontsize=10.5)

ax.text(10.0, 4.1,
        "TD-trained on $(z,\\mathbf{a},r,z')\\sim\\mathcal{D}$,\n"
        "bootstrap actions from $\\mu_\\psi$",
        color="#777777", ha="center", va="center", fontsize=9.5,
        style="italic")

fig.tight_layout()
fig.savefig("fql_pipeline.pdf", bbox_inches="tight")
fig.savefig("fql_pipeline.png", bbox_inches="tight", dpi=200)
print("written fql_pipeline.pdf / fql_pipeline.png")
