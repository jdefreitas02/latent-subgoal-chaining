"""Generate the JEPA vs generative world-model background figure.

Two-panel comparison adapted (in thesis notation) from LeCun (2022) and
Assran et al. (2023): (a) a generative world model decodes predictions
back to pixel space; (b) a joint-embedding predictive world model
predicts the next latent directly and measures error in latent space.

Run:  python make_jepa_figure.py
Outputs: jepa_wm.pdf (for the thesis), jepa_wm.png (preview).
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

IMG_FC, IMG_EC = "#f2f2f2", "#8a8a8a"
ENC_FC, ENC_EC = "#dbe9f9", "#5b8bd0"
DYN_FC, DYN_EC = "#fde6c8", "#d99b46"
DEC_FC, DEC_EC = "#e8dff5", "#9b7fc7"


def box(ax, x, y, w, h, label, fc, ec, fs=13):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.4,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=fs)


def img_node(ax, x, y, label):
    box(ax, x, y, 1.05, 1.05, label, IMG_FC, IMG_EC)


def latent_node(ax, x, y, label):
    ax.add_patch(Circle((x, y), 0.46, fc="white", ec="#3a3a3a", lw=1.4))
    ax.text(x, y, label, ha="center", va="center", fontsize=13)


def arrow(ax, x0, y0, x1, y1, props=ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=props)


def setup(ax, title):
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 6.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13.5, pad=6)


fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.4, 3.7))

# ------------------------------------------------------- (a) generative
setup(axa, "(a) Generative world model")

img_node(axa, 0.9, 4.4, "$s_t$")
box(axa, 2.75, 4.4, 1.5, 0.95, "encoder", ENC_FC, ENC_EC, fs=11.5)
box(axa, 4.95, 4.4, 1.6, 0.95, "dynamics", DYN_FC, DYN_EC, fs=11.5)
box(axa, 7.15, 4.4, 1.5, 0.95, "decoder", DEC_FC, DEC_EC, fs=11.5)
img_node(axa, 9.15, 4.4, r"$\hat{s}_{t+1}$")
img_node(axa, 9.15, 1.35, "$s_{t+1}$")

axa.text(4.95, 6.25, "$a_t$", ha="center", va="center", fontsize=13)
arrow(axa, 4.95, 6.0, 4.95, 5.0)

arrow(axa, 1.5, 4.4, 1.92, 4.4)
arrow(axa, 3.58, 4.4, 4.07, 4.4)
arrow(axa, 5.83, 4.4, 6.32, 4.4)
arrow(axa, 7.98, 4.4, 8.55, 4.4)

arrow(axa, 9.15, 3.78, 9.15, 1.97, LOSS_ARROW)
axa.text(9.55, 2.95, "$\\mathcal{L}_{\\mathrm{rec}}$", color="#c0392b",
         ha="left", va="center", fontsize=13)
axa.text(9.55, 2.35, "(pixel space)", color="#c0392b",
         ha="left", va="center", fontsize=10.5)
axa.text(9.15, 0.42, "ground truth", color="#777777",
         ha="center", va="center", fontsize=10, style="italic")

# ------------------------------------------------- (b) joint-embedding
setup(axb, "(b) Joint-embedding predictive world model")

img_node(axb, 0.95, 4.4, "$s_t$")
box(axb, 2.95, 4.4, 1.45, 0.95, r"$E_\xi$", ENC_FC, ENC_EC)
latent_node(axb, 4.75, 4.4, "$z_t$")
box(axb, 6.45, 4.4, 1.45, 0.95, r"$f_\psi$", DYN_FC, DYN_EC)
latent_node(axb, 8.85, 4.4, r"$\hat{z}_{t+1}$")

img_node(axb, 0.95, 1.35, "$s_{t+1}$")
box(axb, 2.95, 1.35, 1.45, 0.95, r"$E_\xi$", ENC_FC, ENC_EC)
latent_node(axb, 8.85, 1.35, "$z_{t+1}$")

axb.text(6.45, 6.25, "$a_t$", ha="center", va="center", fontsize=13)
arrow(axb, 6.45, 6.0, 6.45, 5.0)

arrow(axb, 1.55, 4.4, 2.15, 4.4)
arrow(axb, 3.75, 4.4, 4.22, 4.4)
arrow(axb, 5.28, 4.4, 5.65, 4.4)
arrow(axb, 7.25, 4.4, 8.32, 4.4)

arrow(axb, 1.55, 1.35, 2.15, 1.35)
arrow(axb, 3.75, 1.35, 8.32, 1.35)

# shared encoder weights
axb.annotate("", xy=(2.95, 1.95), xytext=(2.95, 3.85),
             arrowprops=dict(arrowstyle="-", color="#999999", lw=1.2,
                             linestyle=(0, (2, 2))))
axb.text(2.7, 2.9, "shared", color="#777777", ha="center", va="center",
         fontsize=10, style="italic", rotation=90)

arrow(axb, 8.85, 3.78, 8.85, 1.97, LOSS_ARROW)
axb.text(9.3, 2.95, "$\\mathcal{L}_{\\mathrm{pred}}$", color="#c0392b",
         ha="left", va="center", fontsize=13)
axb.text(9.3, 2.35, "(latent space)", color="#c0392b",
         ha="left", va="center", fontsize=10.5)

axb.text(5.55, 0.32,
         "isotropy regulariser on latents prevents collapse",
         color="#777777", ha="center", va="center", fontsize=10,
         style="italic")

fig.tight_layout()
fig.savefig("jepa_wm.pdf", bbox_inches="tight")
fig.savefig("jepa_wm.png", bbox_inches="tight", dpi=200)
print("written jepa_wm.pdf / jepa_wm.png")
