from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = ROOT / "results"
JSON = RESULTS / "json"
RSCAPE = RESULTS / "rscape"
FIG = ROOT / "figures"


def ensure_runtime_dirs() -> None:
    for d in (DATA, RAW, PROC, RESULTS, JSON, FIG):
        d.mkdir(parents=True, exist_ok=True)


PAL = {
    "obelisk": "#B23A48",
    "obelisk_d": "#7C2230",
    "obelisk_l": "#EFCFD3",
    "delta": "#345E8A",
    "delta_d": "#1F3D5A",
    "delta_l": "#C7D4E2",
    "rfam": "#7E7E7E",
    "shuffle": "#D7D7D7",
    "accent_g": "#3F7C5A",
    "accent_g_l": "#CDDFD2",
    "accent_p": "#7E5B9A",
    "metal": "#B17C2C",
    "basic": "#3D6A92",
    "bg_n": "#FBF6E4",
    "bg_a": "#EDF2F9",
    "bg_c": "#EEF6E9",
    "ink": "#111111",
    "ink_soft": "#4A4A4A",
    "grid": "#EEEEEE",
}

AA_GROUP = {
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "D": "negative",
    "E": "negative",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "C": "polar",
    "F": "aromatic",
    "W": "aromatic",
    "Y": "aromatic",
    "A": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "V": "hydrophobic",
    "G": "flex",
    "P": "flex",
}
GROUP_COL = {
    "positive": PAL["basic"],
    "negative": PAL["obelisk"],
    "polar": PAL["accent_g"],
    "aromatic": "#D77F2B",
    "hydrophobic": PAL["accent_p"],
    "flex": "#9B9B9B",
}

FONT_BODY = 7.0
FONT_TICK = 6.0
FONT_LEGEND = 6.0
FONT_ANNOT = 6.4
FONT_AXLABEL = 7.0
FONT_TITLE = 7.5
FONT_PANEL = 9.0
FONT_FOOT = 5.6

LINE_AX = 0.55
LINE_TICK = 0.55
LINE_DATA = 1.00

WIDTH_DOUBLE = 7.20


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "mathtext.rm": "DejaVu Sans",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": True,
            "font.size": FONT_BODY,
            "axes.titlesize": FONT_TITLE,
            "axes.titleweight": "normal",
            "axes.titlepad": 4.5,
            "axes.titlelocation": "left",
            "axes.labelsize": FONT_AXLABEL,
            "axes.labelpad": 2.4,
            "axes.linewidth": LINE_AX,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#1F1F1F",
            "axes.labelcolor": PAL["ink"],
            "axes.labelweight": "normal",
            "xtick.color": "#1F1F1F",
            "ytick.color": "#1F1F1F",
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "xtick.major.width": LINE_TICK,
            "ytick.major.width": LINE_TICK,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.minor.size": 1.4,
            "ytick.minor.size": 1.4,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": FONT_LEGEND,
            "legend.frameon": False,
            "legend.handlelength": 1.30,
            "legend.handletextpad": 0.50,
            "legend.borderaxespad": 0.5,
            "legend.labelspacing": 0.36,
            "legend.columnspacing": 1.10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 130,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "image.cmap": "viridis",
            "lines.linewidth": LINE_DATA,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.50,
        }
    )


def panel(ax, lbl, x=-0.12, y=1.04, fs=FONT_PANEL):
    ax.text(
        x,
        y,
        lbl,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight="bold",
        va="baseline",
        ha="left",
        color=PAL["ink"],
        family="sans-serif",
    )


def title_left(ax, txt, *, pad=4.5):
    ax.set_title(
        txt, loc="left", pad=pad, fontsize=FONT_TITLE, fontweight="normal", color=PAL["ink"]
    )


def grid(ax, y=True, x=False):
    if y:
        ax.yaxis.grid(True, linestyle="-", linewidth=0.35, color=PAL["grid"], alpha=0.75)
    if x:
        ax.xaxis.grid(True, linestyle="-", linewidth=0.35, color=PAL["grid"], alpha=0.75)
    ax.set_axisbelow(True)


def text_halo(lw=2.0, fg="white"):
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def sig_bar(ax, x1, x2, y, txt, h=0.02, c="#1F1F1F", lw=0.65, fs=FONT_ANNOT):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=c, solid_capstyle="butt")
    ax.text(
        (x1 + x2) / 2,
        y + h * 1.05,
        txt,
        ha="center",
        va="bottom",
        fontsize=fs,
        color=c,
        fontweight="bold",
    )


def polish_fig(fig, max_width=WIDTH_DOUBLE):
    width, height = fig.get_size_inches()
    if width > max_width + 1e-6:
        scale = max_width / width
        fig.set_size_inches(max_width, height * scale, forward=True)
    for ax in fig.axes:
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(LINE_AX)
                spine.set_color("#1F1F1F")
        ax.tick_params(axis="both", which="major", pad=2.0, width=LINE_TICK, length=2.4)
        ax.tick_params(axis="both", which="minor", pad=1.4, width=0.4, length=1.4)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for txt in legend.get_texts():
                txt.set_fontsize(min(txt.get_fontsize(), FONT_LEGEND))
    for legend in fig.legends:
        legend.set_frame_on(False)
        for txt in legend.get_texts():
            txt.set_fontsize(min(txt.get_fontsize(), FONT_LEGEND))


_LOGO_FP = FontProperties(family="sans-serif", weight="bold")


def draw_letter(ax, letter, x, y, w, h, color):
    if h <= 0:
        return
    tp = TextPath((0, 0), letter, size=1, prop=_LOGO_FP)
    bb = tp.get_extents()
    if bb.width <= 0 or bb.height <= 0:
        return
    t = Affine2D().translate(-bb.x0, -bb.y0).scale(w / bb.width, h / bb.height).translate(x, y)
    ax.add_patch(
        PathPatch(
            tp, facecolor=color, edgecolor="none", linewidth=0, transform=t + ax.transData, zorder=4
        )
    )
