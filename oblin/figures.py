
from __future__ import annotations

import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from Bio import SeqIO
from scipy.stats import gaussian_kde
from oblin.core import *
from oblin.core import FIG, JSON, PROC

apply_style()
np.random.seed(42)

OBLIN1 = (
    "MRDIELDSSAFRSQVSLLSQETSEKFLTGAALVSPKRSKYYISEVEGLKVHSRSKKDLLA"
    "LAIISWWLEDSIRFYLQEELYFLSLNNSDLIEIRLCLTSKSGMLNFLEDTTLYHSRDLFG"
    "NILPTSPEKQVRLANLVSVRYGPTSLPKRVIRRRGYKDHGSRRFPHEVHDLSSGKLAQIK"
    "YEEEIQSYHDTLLFLRGWLDGF"
)
N_GLOBE = (1, 130)
DOM_A = (130, 175)
C_HEL = (175, 202)

def _save(fig, name: str) -> None:
    polish_fig(fig)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)

def _safe_load(name: str):
    p = JSON / name
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None

def _load_pdb_ca(path: Path):
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                    continue
                rn = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b = float(line[60:66])
                rows.append((rn, x, y, z, b))
    except OSError:
        return {}
    if not rows:
        return {}
    bmax = max(r[4] for r in rows)
    scale = 100.0 if bmax <= 1.5 else 1.0
    return {rn: (x, y, z, b * scale) for rn, x, y, z, b in rows}

def _info(c, n):
    H = -sum((v / n) * math.log2(v / n) for v in c.values() if v)
    return max(0.0, math.log2(20) - H)

def _missing_panel(ax, msg: str, *, letter: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("#FAFAFA")
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96,
                           transform=ax.transAxes,
                           facecolor="white", edgecolor="#CCCCCC",
                           linewidth=0.6))
    ax.text(0.5, 0.55, "Missing data", ha="center", va="center",
            transform=ax.transAxes, fontsize=FONT_TICK + 1, color="#666666",
            fontstyle="italic")
    ax.text(0.5, 0.42, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=FONT_TICK, color="#888888",
            family="monospace")
    if letter is not None:
        panel(ax, letter)

def fig1_v2():
    print("fig1_v2: BRRC envelope enrichment")
    fc = _safe_load("catalog_features.json")
    env = _safe_load("brrc_envelope_summary.json")
    ci = _safe_load("enrichment_ci.json")
    zs = _safe_load("multi_shuffle_zscores.json")

    fig = plt.figure(figsize=(7.20, 6.20))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.0, 1.10],
        left=0.085, right=0.985, top=0.93, bottom=0.085,
        hspace=0.62, wspace=0.36,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    if not fc or "OBELISK_FULL" not in fc:
        _missing_panel(ax_a, "catalog_features.json missing", letter="A")
    else:
        ob = fc["OBELISK_FULL"]
        sh = fc.get("SHUFFLED_CONTROL", [])
        metrics = [
            ("frac_paired", "Fraction paired", 0.65, ">", "frac. paired"),
            ("sld_per100nt", "Small loops / 100 nt", 10.0, ">", "sld"),
            ("max_loop", "Max loop length (nt)", 15.0, "≤", "max loop"),
        ]
        positions = np.arange(len(metrics)) * 1.6
        for i, (key, label, thr, op, short) in enumerate(metrics):
            ob_v = np.array([r[key] for r in ob], dtype=float)
            sh_v = np.array([r[key] for r in sh], dtype=float) if sh else np.array([])
            ob_v = ob_v[np.isfinite(ob_v)]
            if sh_v.size:
                sh_v = sh_v[np.isfinite(sh_v)]

            if key == "max_loop":
                cap = float(np.percentile(
                    np.concatenate([ob_v, sh_v]) if sh_v.size else ob_v, 99))
                ob_v = np.clip(ob_v, None, cap)
                if sh_v.size:
                    sh_v = np.clip(sh_v, None, cap)

            lo = float(min(ob_v.min(), sh_v.min() if sh_v.size else ob_v.min()))
            hi = float(max(ob_v.max(), sh_v.max() if sh_v.size else ob_v.max()))
            rng_span = hi - lo if hi > lo else 1.0
            ob_n = (ob_v - lo) / rng_span
            sh_n = (sh_v - lo) / rng_span if sh_v.size else sh_v
            thr_n = (thr - lo) / rng_span
            xc = positions[i]

            v_ob = ax_a.violinplot(
                [ob_n], positions=[xc + 0.15], widths=0.62,
                showmeans=False, showmedians=False, showextrema=False)
            for pc_ in v_ob["bodies"]:
                pc_.set_facecolor(PAL["obelisk"])
                pc_.set_alpha(0.78)
                pc_.set_edgecolor(PAL["obelisk_d"])
                pc_.set_linewidth(0.4)
                pts = pc_.get_paths()[0].vertices
                pts[:, 0] = np.clip(pts[:, 0], xc + 0.15, None)

            if sh_v.size:
                v_sh = ax_a.violinplot(
                    [sh_n], positions=[xc - 0.15], widths=0.62,
                    showmeans=False, showmedians=False, showextrema=False)
                for pc_ in v_sh["bodies"]:
                    pc_.set_facecolor(PAL["shuffle"])
                    pc_.set_alpha(0.85)
                    pc_.set_edgecolor("#7E7E7E")
                    pc_.set_linewidth(0.4)
                    pts = pc_.get_paths()[0].vertices
                    pts[:, 0] = np.clip(pts[:, 0], None, xc - 0.15)

            ax_a.plot([xc - 0.42, xc - 0.15], [np.median(sh_n)] * 2,
                      color="#2A2A2A", linewidth=1.05, solid_capstyle="butt"
                      ) if sh_v.size else None
            ax_a.plot([xc + 0.15, xc + 0.42], [np.median(ob_n)] * 2,
                      color=PAL["obelisk_d"], linewidth=1.05,
                      solid_capstyle="butt")

            ax_a.plot([xc - 0.55, xc + 0.55], [thr_n, thr_n],
                      color="#1A1A1A", linestyle=(0, (4.5, 2.4)),
                      linewidth=0.85, alpha=0.85)
            ax_a.text(xc + 0.62, thr_n,
                      f"BRRC {op} {thr:g}",
                      ha="left", va="center", fontsize=FONT_TICK - 0.4,
                      color="#222222",
                      path_effects=text_halo(2.0))

            sig_bar(ax_a, xc - 0.15, xc + 0.15, 1.10, "***",
                    h=0.025, c="#222222", lw=0.95, fs=FONT_ANNOT - 0.3)

            ax_a.text(xc, -0.13, label,
                      ha="center", va="top", fontsize=FONT_AXLABEL,
                      transform=ax_a.get_xaxis_transform(),
                      color="#0F0F0F", fontweight="medium")

        ax_a.set_xticks(positions)
        ax_a.set_xticklabels([])
        ax_a.tick_params(axis="x", length=0)
        ax_a.set_xlim(positions.min() - 0.85, positions.max() + 1.30)
        ax_a.set_ylim(-0.08, 1.30)
        ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_a.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
        ax_a.set_ylabel("Within-criterion normalized rank")
        title_left(ax_a,
                   "Per-criterion split distributions: obelisks (right) vs "
                   "dinucleotide shuffles (left)")

        ax_a.legend(handles=[
            Patch(facecolor=PAL["obelisk"], edgecolor=PAL["obelisk_d"],
                  linewidth=0.4,
                  label=f"Obelisks (n={len(ob):,})"),
            Patch(facecolor=PAL["shuffle"], edgecolor="#7E7E7E",
                  linewidth=0.4,
                  label=f"Dinucleotide shuffles (n={len(sh):,})"),
            Line2D([0], [0], color="#1A1A1A", linestyle=(0, (4.5, 2.4)),
                   linewidth=1.0, label="BRRC threshold"),
        ], loc="upper right", bbox_to_anchor=(1.0, 1.04),
           ncol=3, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
        grid(ax_a)
        panel(ax_a, "A", x=-0.06, y=1.06)

    if not env or "OBELISK" not in env:
        _missing_panel(ax_b, "brrc_envelope_summary.json missing", letter="B")
    else:
        ob_pct = env["OBELISK"]["all_three_pass_pct"]
        sh_pct = env["SHUFFLED"]["all_three_pass_pct"]
        n_ob = env["OBELISK"]["n"]
        n_sh = env["SHUFFLED"]["n"]
        fold = (ci or {}).get(
            "fold_enrichment_point",
            ob_pct / sh_pct if sh_pct > 0 else float("inf"))
        fold_ci = (ci or {}).get(
            "fold_enrichment_Katz_logRR_95_CI", [None, None])

        cats = ["Obelisks", "Dinuc.\nshuffles"]
        vals = [ob_pct, sh_pct]
        cols = [PAL["obelisk"], PAL["shuffle"]]
        xpos = [0, 1]
        ax_b.bar(xpos, vals, color=cols,
                 edgecolor=[PAL["obelisk_d"], "#7E7E7E"],
                 linewidth=0.55, width=0.56, zorder=3)
        ax_b.set_yscale("log")
        ax_b.set_ylim(0.10, 700)
        ax_b.set_yticks([0.1, 1, 10, 100])
        ax_b.set_yticklabels(["0.1", "1", "10", "100"])

        for x, v, n in zip(xpos, vals, [n_ob, n_sh]):
            ax_b.text(x, v * 1.30, f"{v:.2f}%",
                      ha="center", va="bottom",
                      fontsize=FONT_ANNOT, color="#0F0F0F", fontweight="bold")
            ax_b.text(x, v * 2.10, f"n = {n:,}",
                      ha="center", va="bottom",
                      fontsize=FONT_TICK - 0.5, color="#555555")

        ax_b.set_xticks(xpos)
        ax_b.set_xticklabels(cats, fontsize=FONT_AXLABEL)
        ax_b.set_xlim(-0.55, 1.55)
        ax_b.set_ylabel("All-three BRRC pass rate (%, log)")
        title_left(ax_b, "Joint envelope enrichment")

        if math.isfinite(fold):
            try:
                ci_lo, ci_hi = fold_ci or (None, None)
                ci_txt = (f"Katz 95% CI [{ci_lo:.0f}, {ci_hi:.0f}]"
                          if (ci_lo is not None and ci_hi is not None) else "")
            except Exception:
                ci_txt = ""
            br_x = 1.40
            ax_b.plot([br_x - 0.04, br_x, br_x, br_x - 0.04],
                      [sh_pct, sh_pct, ob_pct, ob_pct],
                      color=PAL["obelisk_d"], linewidth=0.95,
                      solid_capstyle="butt", zorder=4)
            ax_b.text(br_x + 0.04, math.sqrt(sh_pct * ob_pct),
                      f"{fold:.0f}×",
                      ha="left", va="center",
                      fontsize=FONT_TITLE, color=PAL["obelisk_d"],
                      fontweight="bold",
                      path_effects=text_halo(2.4))
            ax_b.text(br_x + 0.04, math.sqrt(sh_pct * ob_pct) * 0.35,
                      ci_txt,
                      ha="left", va="center",
                      fontsize=FONT_TICK - 0.8, color=PAL["obelisk_d"],
                      fontstyle="italic",
                      path_effects=text_halo(1.8))
        ax_b.set_xlim(-0.55, 1.85)

        grid(ax_b)
        panel(ax_b, "B", x=-0.15, y=1.06)

    if not zs:
        _missing_panel(ax_c, "multi_shuffle_zscores.json missing", letter="C")
    else:
        zf = np.array([r["z_frac_paired"] for r in zs])
        zs_ = np.array([r["z_small_loop_density_per100nt"] for r in zs])
        n_both = int(np.sum((zf > 2) & (zs_ > 2)))
        ax_c.axvspan(2, zf.max() * 1.10, color=PAL["accent_g"], alpha=0.07, zorder=0)
        ax_c.axhspan(2, zs_.max() * 1.10, color=PAL["accent_g"], alpha=0.07, zorder=0)
        ax_c.scatter(zf, zs_, s=6, c=PAL["obelisk"], alpha=0.32,
                     edgecolor="none", zorder=3)
        ax_c.axhline(0, color="#9B9B9B", linewidth=0.5, zorder=1)
        ax_c.axvline(0, color="#9B9B9B", linewidth=0.5, zorder=1)
        ax_c.axhline(2, color="#444444", linestyle=(0, (4, 2.5)),
                     linewidth=0.85, zorder=2)
        ax_c.axvline(2, color="#444444", linestyle=(0, (4, 2.5)),
                     linewidth=0.85, zorder=2)

        ax_c.set_xlabel(r"Per-sequence $z$ (fraction paired)")
        ax_c.set_ylabel(r"Per-sequence $z$ (small-loop density)")
        title_left(ax_c, "Per-sequence z-scores vs 10 shuffles each")

        xlim = (min(-1.0, zf.min() * 1.05), max(8.0, zf.max() * 1.05))
        ylim = (min(-1.0, zs_.min() * 1.05), max(8.0, zs_.max() * 1.05))
        ax_c.set_xlim(*xlim)
        ax_c.set_ylim(*ylim)

        ax_c.text(2.20, ylim[0] + 0.15, r"$z = +2$",
                  ha="left", va="bottom",
                  fontsize=FONT_TICK - 0.4, color="#444444",
                  path_effects=text_halo(1.6))
        ax_c.text(xlim[0] + 0.15, 2.20, r"$z = +2$",
                  ha="left", va="bottom",
                  fontsize=FONT_TICK - 0.4, color="#444444",
                  path_effects=text_halo(1.6))

        grid(ax_c)
        panel(ax_c, "C", x=-0.17, y=1.06)

    _save(fig, "Figure1_BRRC")

def fig2_v2():
    print("fig2_v2: robustness + external validation")
    boot = _safe_load("cluster_level_bootstrap.json")
    hsob = _safe_load("hsob_validation.json")
    hi_raw = _safe_load("higher_order_nulls.json")
    hi = None
    if hi_raw:
        by = hi_raw.get("by_null", {})
        def _row(d, n_total):
            ci = d.get("wilson_95ci_pct", [None, None])
            return {"k": d.get("n_pass"), "n": d.get("n_total_shuffles", n_total),
                    "pct": d.get("pass_pct"), "ci_low": ci[0], "ci_high": ci[1]}
        n_total = 15507
        hi = {
            "trinuc_eulerian": _row(by.get("trinuc", {}), n_total),
            "markov3":         _row(by.get("markov3", {}), n_total),
            "syn_codon_freq":  _row(by.get("synonymous_codon", {}), n_total),
        }
    lf = _safe_load("lf_concord.json")
    env = _safe_load("brrc_envelope_summary.json")
    ens = _safe_load("ensemble_brrc.json")
    lf_match = _safe_load("linearfold_shuffles.json")

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 7.20))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.085,
                        hspace=0.62, wspace=0.35)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    if not boot:
        _missing_panel(ax_a, "cluster_level_bootstrap.json missing", letter="A")
    else:
        loo = [r for r in boot.get("leave_one_cluster_out", [])
               if isinstance(r, dict) and "equal_weight_pct" in r]
        loo_sorted = sorted(loo, key=lambda r: r["equal_weight_pct"])
        vals = [r["equal_weight_pct"] for r in loo_sorted]
        names = [r.get("left_out", "?") for r in loo_sorted]
        obs = boot.get("obs_eq_weight_pct")
        ci = boot.get("equal_weight_bootstrap_95_CI", [None, None])

        if ci[0] is not None:
            ax_a.axhspan(ci[0], ci[1], color=PAL["accent_g"], alpha=0.10,
                         zorder=0)
            for y_edge in (ci[0], ci[1]):
                ax_a.axhline(y_edge, color=PAL["accent_g"], linewidth=0.55,
                             alpha=0.7, zorder=0.3)

        if obs is not None:
            ax_a.axhline(obs, color=PAL["obelisk"], linewidth=1.1,
                         zorder=2)
        x = np.arange(1, len(vals) + 1)
        ax_a.scatter(x, vals, s=22, color=PAL["delta"],
                     edgecolor="white", linewidth=0.50, zorder=3)

        if vals:
            med = float(np.median(vals))
            order = sorted(range(len(vals)),
                           key=lambda i: -abs(vals[i] - med))
            outliers = [i for i in order if abs(vals[i] - med) > 0.4][:3]
            outliers_sorted = sorted(outliers, key=lambda i: x[i])
            placed = []
            n_loco = len(vals)
            for i in outliers_sorted:
                xa, ya = x[i], vals[i]
                above = ya >= med
                dy = 1.8 if above else -1.8
                ha_ = "right" if xa > n_loco * 0.55 else "left"
                dx = -0.6 if ha_ == "right" else 0.6
                tx = xa + dx
                ty = ya + dy
                while any(abs(tx - xp) < 1.2 and abs(ty - yp) < 1.0
                          for xp, yp in placed):
                    ty += dy * 0.5
                ax_a.annotate(
                    names[i], xy=(xa, ya), xytext=(tx, ty),
                    textcoords="data",
                    ha=ha_, va="bottom" if above else "top",
                    fontsize=FONT_TICK - 0.3, color="#333333",
                    fontstyle="italic", fontweight="medium",
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                    linewidth=0.55, shrinkA=0, shrinkB=4),
                    path_effects=text_halo(1.8))
                placed.append((tx, ty))

            ax_a.set_ylim(min(min(vals), ci[0] or min(vals)) - 1.4,
                          max(max(vals), ci[1] or max(vals)) + 5.0)
        ax_a.set_xlim(0.4, len(vals) + 0.6)
        ax_a.set_xlabel("LOCO replicate (sorted by pass rate)")
        ax_a.set_ylabel("Leave-one-cluster-out pass rate (%)")
        title_left(ax_a,
                   f"Cluster-level robustness "
                   f"(n = {boot.get('n_clusters', '-')} clusters)")

        ax_a.legend(handles=[
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=PAL["delta"], markeredgecolor="white",
                   markersize=4.4, markeredgewidth=0.5,
                   label="LOCO replicate"),
            Line2D([0], [0], color=PAL["obelisk"], linewidth=1.4,
                   label=f"Observed = {obs:.1f}%"),
            Patch(facecolor=PAL["accent_g"], alpha=0.20,
                  label=f"95% CI [{ci[0]:.1f}, {ci[1]:.1f}]"),
        ], loc="upper left", bbox_to_anchor=(0.0, 1.0),
           ncol=1, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, labelspacing=0.30,
           borderaxespad=0.4)
        grid(ax_a)
        panel(ax_a, "A", x=-0.14, y=1.06)

    if not hsob:
        _missing_panel(ax_b, "hsob_validation.json missing", letter="B")
    else:
        import re
        def _ck(s):
            m = re.match(r"([A-Za-z]+)(\d+)", s)
            return (m.group(1), int(m.group(2))) if m else (s, 0)
        by = hsob["by_clade"]
        items = sorted(by.items(), key=lambda kv: _ck(kv[0]))
        clades = [k for k, _ in items]
        rates = [100 * by[k]["fp_pass_rate"] for k in clades]
        x = np.arange(len(clades))

        ax_b.bar(x, rates, color=PAL["delta"], edgecolor=PAL["delta_l"],
                 linewidth=0.55, width=0.78, zorder=3)
        overall = 100 * hsob.get("frac_paired_pass_rate", 0.0)
        n_total = hsob.get("HsOb_n_total", "-")

        ax_b.axhline(overall, color=PAL["obelisk"], linewidth=1.25,
                     linestyle=(0, (4, 2.4)), zorder=4)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(clades, rotation=55, ha="right",
                             rotation_mode="anchor",
                             fontsize=FONT_TICK - 1.0)
        ax_b.tick_params(axis="x", pad=1.0)
        ax_b.set_ylabel(r"HsOb fraction paired $>$ 0.65 (%)")
        ax_b.set_ylim(0, 122)
        title_left(ax_b, "HsOb 2026 held-out validation by clade")

        try:
            n_total_txt = f"{n_total:,}"
        except (TypeError, ValueError):
            n_total_txt = str(n_total)

        ax_b.text(len(clades) - 0.5, overall + 12,
                  f"Overall = {overall:.1f}% (n = {n_total_txt})",
                  ha="right", va="bottom",
                  fontsize=FONT_TICK, color=PAL["obelisk"],
                  fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25",
                            facecolor="white", edgecolor="none",
                            alpha=0.85))

        ax_b.legend(handles=[
            Patch(facecolor=PAL["delta"], edgecolor=PAL["delta_l"],
                  linewidth=0.5, label="HsOb clade"),
            Line2D([0], [0], color=PAL["obelisk"],
                   linestyle=(0, (4, 2.4)), linewidth=1.4,
                   label="Overall pass"),
        ], loc="upper left", bbox_to_anchor=(0.0, 1.0),
           ncol=1, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, labelspacing=0.30,
           borderaxespad=0.4)
        grid(ax_b)
        panel(ax_b, "B", x=-0.14, y=1.06)

    if not hi:
        _missing_panel(ax_c, "higher_order_nulls.json missing", letter="C")
    else:
        def _wilson95(k: int, n: int) -> tuple[float, float]:
            if not n:
                return (0.0, 0.0)
            z = 1.959963984540054
            p = k / n
            denom = 1.0 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            spread = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
                      / denom)
            return (max(0.0, (center - spread) * 100.0),
                    min(100.0, (center + spread) * 100.0))

        bars_x = []
        bars_y = []
        bar_cols = []
        ci_lo: list[float | None] = []
        ci_hi: list[float | None] = []
        ns: list[int | None] = []
        labels_above = []
        if env:
            ob_pct = env["OBELISK"]["all_three_pass_pct"]
            ob_n = env["OBELISK"].get("n", 0)
            ob_k = env["OBELISK"].get("all_three_pass", 0)
            ob_lo, ob_hi = _wilson95(ob_k, ob_n)
            bars_x.append("Obelisks")
            bars_y.append(ob_pct)
            bar_cols.append(PAL["obelisk"])
            ci_lo.append(ob_lo); ci_hi.append(ob_hi)
            ns.append(ob_n)
            labels_above.append(f"{ob_pct:.1f}%")
            sh_pct = env["SHUFFLED"]["all_three_pass_pct"]
            sh_n = env["SHUFFLED"].get("n", 0)
            sh_k = env["SHUFFLED"].get("all_three_pass", 0)
            sh_lo, sh_hi = _wilson95(sh_k, sh_n)
            bars_x.append("Dinuc.")
            bars_y.append(sh_pct)
            bar_cols.append(PAL["shuffle"])
            ci_lo.append(sh_lo); ci_hi.append(sh_hi)
            ns.append(sh_n)
            labels_above.append(f"{sh_pct:.2f}%")
        for nm, lbl in [("trinuc_eulerian", "Tri-Eulerian"),
                        ("markov3", "Markov-3"),
                        ("syn_codon_freq", "Syn-codon")]:
            row = hi.get(nm, {})
            pct = row.get("pct")
            if pct is None:
                continue
            bars_x.append(lbl)
            bars_y.append(pct)
            bar_cols.append("#9D9D9D")
            lo = row.get("ci_low", row.get("wilson_lo"))
            up = row.get("ci_high", row.get("wilson_hi"))
            ci_lo.append(lo); ci_hi.append(up)
            ns.append(row.get("n"))
            labels_above.append(f"{pct:.1f}%")

        x = np.arange(len(bars_x))
        edges = ["#7C2230", "#7E7E7E"] + ["#4E4E4E"] * (len(bars_x) - 2)
        ax_c.bar(x, bars_y, color=bar_cols, edgecolor=edges,
                 linewidth=0.55, width=0.65, zorder=3)
        ax_c.set_yscale("log")
        ax_c.set_ylim(0.08, 320)

        err_x = []; err_y = []; err_lo = []; err_hi = []
        per_bar_top = []
        for xi, y, lo, up in zip(x, bars_y, ci_lo, ci_hi):
            if lo is None or up is None:
                per_bar_top.append(y)
                continue
            err_x.append(xi)
            err_y.append(y)
            err_lo.append(max(y - lo, 1e-3))
            err_hi.append(max(up - y, 1e-3))
            per_bar_top.append(max(y, up))
        if err_x:
            ax_c.errorbar(err_x, err_y, yerr=[err_lo, err_hi],
                          fmt="none", ecolor="#2C2C2C", elinewidth=1.15,
                          capsize=4.0, capthick=1.15, zorder=4)

        for xi, y, lbl, top in zip(x, bars_y, labels_above, per_bar_top):
            ax_c.text(xi, top * 1.75, lbl, ha="center", va="bottom",
                      fontsize=FONT_TICK - 0.3, color="#0F0F0F",
                      fontweight="bold")

        ax_c.set_xticks(x)
        ax_c.set_xticklabels(bars_x, fontsize=FONT_TICK - 0.4,
                             rotation=18, ha="right")
        ax_c.set_ylabel("All-three BRRC pass rate (%, log)")
        title_left(ax_c, "Survival under increasingly stringent nulls")

        ax_c.legend(handles=[
            Patch(facecolor=PAL["obelisk"], edgecolor=PAL["obelisk_d"],
                  linewidth=0.5, label="Obelisks"),
            Patch(facecolor=PAL["shuffle"], edgecolor="#7E7E7E",
                  linewidth=0.5, label="Dinuc."),
            Patch(facecolor="#9D9D9D", edgecolor="#4E4E4E",
                  linewidth=0.5, label="Higher-order nulls"),
            Line2D([0], [0], color="#2C2C2C", marker="_", linestyle="none",
                   markersize=6, label="95% CI"),
        ], loc="upper right", bbox_to_anchor=(1.0, 1.0),
           ncol=1, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, labelspacing=0.30,
           borderaxespad=0.4)
        grid(ax_c)
        panel(ax_c, "C", x=-0.14, y=1.06)

    if not lf:
        _missing_panel(ax_d, "lf_concord.json missing", letter="D")
    else:
        engines = ["ViennaRNA", "LinearFold\nViennaRNA params", "LinearFold\nCONTRAfold params"]
        rates = [lf.get("vrna_pass_pct"),
                 lf.get("lf_vienna_pct"),
                 lf.get("lf_contrafold_pct")]
        concord = [None,
                   lf.get("vrna_vs_lfv_pct"),
                   lf.get("vrna_vs_lfc_pct")]
        cols = [PAL["obelisk"], PAL["delta"], PAL["accent_p"]]
        edges = [PAL["obelisk_d"], "#1F3A52", "#48356E"]
        x = np.arange(len(engines))
        ax_d.bar(x, rates, color=cols, edgecolor=edges,
                 linewidth=0.55, width=0.56, zorder=3)
        for xi, v in zip(x, rates):
            if v is None:
                continue
            ax_d.text(xi, v + 3.0, f"{v:.0f}%",
                      ha="center", va="bottom",
                      fontsize=FONT_ANNOT, color="#0F0F0F", fontweight="bold",
                      path_effects=text_halo(1.8))

        cf_enrich_lo = None
        if lf_match is not None:
            er_lo = lf_match.get("enrichment_katz95_lo")
            if er_lo is None and lf_match.get(
                    "rule_of_three_pct"):
                native = lf_match.get("native_pass_pct", 0)
                upper = lf_match.get("rule_of_three_pct")
                if upper and upper > 0:
                    cf_enrich_lo = native / upper
            else:
                cf_enrich_lo = er_lo

        sub_labels = []
        for eng, c in zip(engines, concord):
            if c is None:
                sub_labels.append("reference")
            elif "CONTRAfold" in eng:
                if cf_enrich_lo is not None and cf_enrich_lo >= 1:
                    sub_labels.append(rf"$\geq${cf_enrich_lo:.0f}× vs null")
                else:
                    sub_labels.append("CONTRAfold params")
            else:
                sub_labels.append(f"{c:.0f}% concord.")

        ax_d.set_xticks(x)
        tick_labels = [f"{eng}\n({sub})" for eng, sub in zip(engines, sub_labels)]
        ax_d.set_xticklabels(tick_labels, fontsize=FONT_TICK - 1.0,
                             linespacing=1.20)
        ax_d.tick_params(axis="x", pad=1.0)
        ax_d.set_ylabel("BRRC pass rate (%)")
        title_left(ax_d, "Independent-engine concordance")

        ax_d.set_ylim(0, max(filter(None, rates)) * 1.32)

        grid(ax_d)
        panel(ax_d, "D", x=-0.14, y=1.06)

    _save(fig, "Figure2_robustness")

def fig3_v2():
    print("fig3_v2: Oblin-1 conservation")
    fig = plt.figure(figsize=(7.20, 6.20))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.10, 0.95],
        width_ratios=[1.0, 1.0],
        hspace=0.62, wspace=0.34,
        left=0.085, right=0.985, top=0.93, bottom=0.085,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    try:
        from Bio import SeqIO
        seqs = [str(r.seq)
                for r in SeqIO.parse(PROC / "domain_A_alignment.fasta", "fasta")
                if len(str(r.seq)) == 17]
    except Exception:
        seqs = []
    if len(seqs) < 5:
        _missing_panel(ax_a, "data/processed/domain_A_alignment.fasta",
                       letter="A")
    else:
        L = 17
        n = len(seqs)
        counts = [Counter(s[i] for s in seqs) for i in range(L)]
        bits = [_info(c, n) for c in counts]
        top_frac = [counts[i].most_common(1)[0][1] / n for i in range(L)]

        ax_a.set_xlim(0.5, L + 0.5)
        ax_a.set_ylim(0, 5.0)
        ax_a.set_xticks(range(1, L + 1))
        ax_a.set_xticklabels(range(1, L + 1), fontsize=FONT_TICK - 0.4)
        ax_a.set_yticks([0, 1, 2, 3, 4])
        ax_a.set_ylabel("Information (bits)")
        title_left(ax_a, f"Domain-A sequence logo (n = {n:,} obelisks)")

        for pos_i in range(L):
            items = sorted(counts[pos_i].items(), key=lambda kv: kv[1])
            y_cur = 0.0
            for letter, cnt in items:
                h = bits[pos_i] * (cnt / n)
                if h < 0.20:
                    y_cur += h
                    continue
                col = GROUP_COL.get(AA_GROUP.get(letter, "flex"), "#999999")
                draw_letter(ax_a, letter, pos_i + 1 - 0.36, y_cur + 0.02,
                            0.72, max(h - 0.05, 0.01), col)
                y_cur += h

        annotated_positions = [(6, "G6"), (7, "Y7"), (9, "D9"),
                               (11, "G11"), (17, "H17")]
        caret_gap = 0.26
        label_gap = 0.30
        for p, lbl in annotated_positions:
            if top_frac[p - 1] >= 0.83:
                top_y = bits[p - 1]
                caret_y = top_y + caret_gap
                label_y = caret_y + label_gap
                ax_a.plot([p], [caret_y], marker="v",
                          color=PAL["obelisk_d"],
                          markersize=4.5, markeredgecolor="white",
                          markeredgewidth=0.4, zorder=6)
                ax_a.text(p, label_y, lbl,
                          ha="center", va="bottom",
                          fontsize=FONT_TICK - 0.4, fontweight="bold",
                          color=PAL["obelisk_d"],
                          path_effects=text_halo(1.8))
        ax_a.set_xlabel("Domain-A position")
        legend_handles = [
            Patch(facecolor=GROUP_COL["positive"], label="basic (K/R/H)"),
            Patch(facecolor=GROUP_COL["negative"], label="acidic (D/E)"),
            Patch(facecolor=GROUP_COL["polar"], label="polar"),
            Patch(facecolor=GROUP_COL["aromatic"], label="aromatic"),
            Patch(facecolor=GROUP_COL["hydrophobic"], label="hydrophobic"),
            Patch(facecolor=GROUP_COL["flex"], label="flex (G/P)"),
        ]
        ax_a.legend(handles=legend_handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=6, fontsize=FONT_LEGEND - 0.7,
                    handletextpad=0.4, columnspacing=1.4,
                    borderaxespad=0.0)
        grid(ax_a)
        panel(ax_a, "A", x=-0.05, y=1.07)

    cons = _safe_load("oblin1_conservation.json")
    if not cons:
        _missing_panel(ax_b, "oblin1_conservation.json missing", letter="B")
    else:
        pos_frac = np.array([r["pos_frac"] for r in cons], dtype=float)
        bins = np.linspace(0, 0.30, 32)
        n_counts, _, _ = ax_b.hist(pos_frac, bins=bins, color=PAL["basic"],
                                   alpha=0.78, edgecolor="white",
                                   linewidth=0.30, zorder=3)
        mn = float(pos_frac.mean())
        ref_pos = sum(1 for a in OBLIN1 if a in "KR") / len(OBLIN1)
        ax_b.set_xlim(0, 0.30)

        bar_max = float(n_counts.max()) if len(n_counts) else 1.0
        ax_b.set_ylim(0, bar_max * 1.78)
        line_top = bar_max * 1.10

        ax_b.plot([mn, mn], [0, line_top], color="#1A1A1A",
                  linestyle=(0, (4, 2.4)), linewidth=0.95, zorder=4)
        ax_b.plot([ref_pos, ref_pos], [0, line_top], color=PAL["obelisk"],
                  linewidth=1.10, zorder=4)

        markers = sorted([
            (0.078,    "Hfq",          "#7A7A7A"),
            (ref_pos,  "Oblin-1 ref.", PAL["obelisk"]),
            (mn,       "catalog mean", "#1A1A1A"),
            (0.156,    "HU-A",         PAL["accent_p"]),
            (0.164,    "StpA",         PAL["accent_g"]),
        ], key=lambda m: m[0])
        marker_y = bar_max * 1.18
        label_y = bar_max * 1.50
        x_lo, x_hi = 0.025, 0.275
        n_m = len(markers)
        target_xs = np.linspace(x_lo, x_hi, n_m)
        for (xv, lbl, col), tx in zip(markers, target_xs):
            ax_b.plot([xv, xv], [0, marker_y - bar_max * 0.04],
                      color=col, linewidth=0.7,
                      linestyle=":", alpha=0.85, zorder=4)
            ax_b.plot([xv], [marker_y - bar_max * 0.04], marker="v",
                      color=col, markersize=4.5,
                      markeredgecolor="white", markeredgewidth=0.4,
                      zorder=5)
            ax_b.plot([xv, tx], [marker_y, label_y - bar_max * 0.03],
                      color=col, linewidth=0.45, alpha=0.85, zorder=4)
            value = f"{xv:.3f}"
            ax_b.text(tx, label_y, f"{lbl}\n{value}",
                      ha="center", va="bottom",
                      fontsize=FONT_TICK - 0.5, color=col,
                      linespacing=1.10,
                      path_effects=text_halo(2.2))

        ax_b.set_xlabel("Basic-residue fraction (K + R)")
        ax_b.set_ylabel(f"# Oblin-1 proteins (n = {len(pos_frac):,})")
        title_left(ax_b, "Basic composition vs nucleic-acid binders")

        grid(ax_b)
        panel(ax_b, "B", x=-0.14, y=1.06)

    cov = _safe_load("covariation_summary.json")
    if not cov:
        _missing_panel(ax_c, "covariation_summary.json missing", letter="C")
    else:
        msa_keys = ["omicron", "pi", "rho"]
        x = np.arange(len(msa_keys))
        pct_canon = []
        pct_comp = []
        labels = []
        for k in msa_keys:
            row = (cov or {}).get(k, {})
            s = row.get("summary", {})
            pct_canon.append(s.get("pct_canon_preserved", 0))
            pct_comp.append(s.get("pct_compensatory_evidence", 0))
            labels.append(f"{k}\n(n={row.get('n_seqs_used', '-')})")
        w = 0.38
        ax_c.bar(x - w / 2, pct_canon, w, color=PAL["delta"],
                 edgecolor="#1F3A52", linewidth=0.55,
                 label=r"% canonical preserved")
        ax_c.bar(x + w / 2, pct_comp, w, color=PAL["accent_p"],
                 edgecolor="#48356E", linewidth=0.55,
                 label="% APC-MI compensatory (descriptive)")
        for xi, v in zip(x - w / 2, pct_canon):
            ax_c.text(xi, v + 2.5, f"{v:.0f}%", ha="center", va="bottom",
                      fontsize=FONT_TICK, color="#0F0F0F", fontweight="bold")
        for xi, v in zip(x + w / 2, pct_comp):
            ax_c.text(xi, v + 2.5, f"{v:.0f}%", ha="center", va="bottom",
                      fontsize=FONT_TICK, color="#0F0F0F", fontweight="bold")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(labels, fontsize=FONT_TICK - 0.4)
        ax_c.set_ylabel("% base pairs (per subfamily MSA)")
        ax_c.set_ylim(0, 155)
        title_left(ax_c, "Within-subfamily covariation")
        ax_c.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                    ncol=1, fontsize=FONT_LEGEND - 0.7,
                    handletextpad=0.4, labelspacing=0.32,
                    borderaxespad=0.4)
        grid(ax_c)
        panel(ax_c, "C", x=-0.14, y=1.06)

    _save(fig, "Figure3_oblin1_conservation")

def fig4_v2():
    print("fig4_v2: Oblin-1 structure + homology negatives")
    fig = plt.figure(figsize=(7.20, 7.20))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.0],
        width_ratios=[1.15, 1.0],
        hspace=0.55, wspace=0.34,
        left=0.085, right=0.985, top=0.93, bottom=0.085,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    ca_af3 = _load_pdb_ca(PROC / "oblin1_af3.pdb")
    ca_boltz = _load_pdb_ca(PROC / "oblin1_boltz2.pdb")
    ca_esm = _load_pdb_ca(PROC / "oblin1_esmfold.pdb")

    if not (ca_af3 and ca_esm):
        _missing_panel(ax_a, "data/processed/oblin1_*.pdb missing", letter="A")
    else:
        pos = sorted(set(ca_af3) | set(ca_esm) | set(ca_boltz))
        plddt_af3 = np.array([ca_af3.get(p, (0, 0, 0, 0))[3] for p in pos])
        plddt_boltz = np.array([ca_boltz.get(p, (0, 0, 0, 0))[3] for p in pos])
        plddt_esm = np.array([ca_esm.get(p, (0, 0, 0, 0))[3] for p in pos])

        ax_a.axvspan(*N_GLOBE, alpha=0.60, color=PAL["bg_n"], zorder=0)
        ax_a.axvspan(*DOM_A,   alpha=0.60, color=PAL["bg_a"], zorder=0)
        ax_a.axvspan(*C_HEL,   alpha=0.60, color=PAL["bg_c"], zorder=0)

        for ((a0, a1), lbl, col) in (
                (N_GLOBE, "N-globule",  "#8E7C3E"),
                (DOM_A,   "Domain-A",   "#3D5E83"),
                (C_HEL,   "C-helix",    "#3F6B47")):
            ax_a.text(0.5 * (a0 + a1), 106.0, lbl,
                      ha="center", va="center",
                      fontsize=FONT_TICK, fontweight="semibold",
                      color=col, path_effects=text_halo(2.2))

        ax_a.plot(pos, plddt_esm, color="#9B9B9B", linewidth=1.10,
                  alpha=0.92, zorder=3)
        ax_a.plot(pos, plddt_boltz, color=PAL["delta"], linewidth=1.25,
                  zorder=4)
        ax_a.plot(pos, plddt_af3, color=PAL["obelisk"], linewidth=1.35,
                  zorder=5)

        ax_a.axhline(70, color="#1A1A1A", linestyle=(0, (4, 2.4)),
                     linewidth=0.85, alpha=0.85)
        ax_a.set_xlim(min(pos), max(pos))
        ax_a.set_ylim(0, 115)
        ax_a.set_yticks([0, 25, 50, 70, 100])
        ax_a.text(max(pos) - 4, 78, "pLDDT 70 (confident)",
                  ha="right", va="bottom",
                  fontsize=FONT_TICK - 0.2, color="#222222",
                  path_effects=text_halo(2.4))
        ax_a.set_xlabel("Residue position")
        ax_a.set_ylabel("pLDDT")
        title_left(ax_a,
                   "Per-residue pLDDT: AF3 + Boltz-2 share an MSA, "
                   "ESMFold is MSA-free")

        ax_a.legend(handles=[
            Line2D([0], [0], color=PAL["obelisk"], linewidth=1.6,
                   label=f"AlphaFold3 (mean {plddt_af3.mean():.1f})"),
            Line2D([0], [0], color=PAL["delta"], linewidth=1.5,
                   label=f"Boltz-2 (mean {plddt_boltz.mean():.1f})"),
            Line2D([0], [0], color="#9B9B9B", linewidth=1.3,
                   label=f"ESMFold (mean {plddt_esm.mean():.1f})"),
        ], loc="lower right", bbox_to_anchor=(1.0, 0.04),
           ncol=3, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, columnspacing=1.2,
           borderaxespad=0.2)
        grid(ax_a)
        panel(ax_a, "A", x=-0.06, y=1.06)

    if not (ca_af3 and ca_boltz):
        _missing_panel(ax_b, "AF3 / Boltz PDB missing", letter="B")
    else:
        common = sorted(set(ca_af3) & set(ca_boltz))
        af = np.array([ca_af3[p][:3] for p in common])
        bz = np.array([ca_boltz[p][:3] for p in common])
        af = af - af.mean(axis=0)
        bz = bz - bz.mean(axis=0)

        ax_b.plot(af[:, 0], af[:, 2], color=PAL["obelisk"], linewidth=1.50,
                  alpha=1.0, zorder=4)
        ax_b.plot(bz[:, 0], bz[:, 2], color=PAL["delta_d"], linewidth=1.50,
                  alpha=0.95, linestyle=(0, (4.5, 2.4)), zorder=3)

        markers = [
            (158, "D158", ( 30,  24), "left",  "bottom"),
            (159, "H159", ( 30, -24), "left",  "top"),
            (166, "H166", (-30, -28), "right", "top"),
        ]
        for p, lbl, offset, ha_, va_ in markers:
            if p in ca_af3:
                xy = af[common.index(p)]
                ax_b.scatter(xy[0], xy[2], s=110, marker="*",
                             color=PAL["metal"], edgecolor="black",
                             linewidth=0.75, zorder=10)
                ax_b.annotate(
                    lbl, (xy[0], xy[2]),
                    xytext=offset, textcoords="offset points",
                    fontsize=FONT_TICK + 0.4, fontweight="bold",
                    color=PAL["metal"], zorder=11,
                    ha=ha_, va=va_,
                    path_effects=text_halo(2.8),
                    arrowprops=dict(arrowstyle="-", color=PAL["metal"],
                                    linewidth=1.4, alpha=1.0,
                                    shrinkA=3, shrinkB=5))

        xs = np.concatenate([af[:, 0], bz[:, 0]])
        zs = np.concatenate([af[:, 2], bz[:, 2]])
        ax_b.set_aspect("equal")
        ax_b.set_xlim(xs.min() - 6, xs.max() + 14)
        ax_b.set_ylim(zs.min() - 8, zs.max() + 8)
        ax_b.set_xlabel(r"X (Å)")
        ax_b.set_ylabel(r"Z (Å)")
        title_left(ax_b, r"AF3 vs Boltz-2 C$_\alpha$ overlay (XZ projection)")

        ax_b.legend(handles=[
            Line2D([0], [0], color=PAL["obelisk"], linewidth=1.8,
                   label="AlphaFold3"),
            Line2D([0], [0], color=PAL["delta_d"], linewidth=1.8,
                   linestyle=(0, (4.5, 2.4)), label="Boltz-2"),
            Line2D([0], [0], marker="*", linestyle="none",
                   markerfacecolor=PAL["metal"], markeredgecolor="black",
                   markersize=8.0, label="Domain-A residues"),
        ], loc="upper right", bbox_to_anchor=(1.0, 1.0),
           ncol=1, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, labelspacing=0.30,
           borderaxespad=0.4)

        grid(ax_b, y=False)
        panel(ax_b, "B", x=-0.18, y=1.06)

    fs_af = _safe_load("oblin1_foldseek_af3.json")
    fs_bz = _safe_load("oblin1_foldseek_boltz2.json")
    if not fs_af:
        _missing_panel(ax_c, "oblin1_foldseek_af3.json missing", letter="C")
    else:
        dbs = list(fs_af.get("per_db", {}).keys())
        pretty = {"afdb50": "AFDB50", "cath50": "CATH50",
                  "gmgcl_id": "GMGCL", "mgnify_esm30": "MGnify-ESM30",
                  "pdb100": "PDB100"}
        labels = [pretty.get(k, k) for k in dbs]

        def _eval_or_nan(rec):
            v = (rec.get("top_hit") or {}).get("eval") if rec else None
            return np.nan if v is None else v

        e_af = [_eval_or_nan(fs_af["per_db"].get(k)) for k in dbs]
        if fs_bz:
            e_bz = [_eval_or_nan(fs_bz.get("per_db", {}).get(k)) for k in dbs]
        else:
            e_bz = [np.nan] * len(dbs)
        y = np.arange(len(dbs))
        h = 0.38

        ax_c.barh(y - h / 2, e_af, h, color=PAL["obelisk"],
                  edgecolor=PAL["obelisk_d"], linewidth=0.55,
                  label="AlphaFold3 input")
        ax_c.barh(y + h / 2, e_bz, h, color=PAL["delta"],
                  edgecolor="#1F3A52", linewidth=0.55,
                  label="Boltz-2 input")

        for yi, ea in zip(y - h / 2, e_af):
            if np.isfinite(ea):
                ax_c.text(ea * 1.22, yi, f"E = {ea:.2g}",
                          va="center", ha="left",
                          fontsize=FONT_TICK - 0.5, color="#0F0F0F",
                          path_effects=text_halo(2.0))
        for yi, eb in zip(y + h / 2, e_bz):
            if np.isfinite(eb):
                ax_c.text(eb * 1.22, yi, f"E = {eb:.2g}",
                          va="center", ha="left",
                          fontsize=FONT_TICK - 0.5, color="#0F0F0F",
                          path_effects=text_halo(2.0))

        ax_c.axvline(0.05, color=PAL["accent_g"], linestyle=(0, (4, 2.4)),
                     linewidth=1.0, zorder=4)
        ax_c.text(0.054, ax_c.get_ylim()[1] - 0.20, "E = 0.05",
                  ha="left", va="top",
                  fontsize=FONT_TICK - 0.4, color=PAL["accent_g"],
                  fontweight="bold",
                  path_effects=text_halo(2.0))

        ax_c.set_yticks(y)
        ax_c.set_yticklabels(labels, fontsize=FONT_TICK - 0.2)
        ax_c.invert_yaxis()
        ax_c.set_xscale("log")
        ax_c.set_xlim(0.03, 600)
        ax_c.set_xlabel("Foldseek best-hit E-value (log scale)")
        title_left(ax_c, "No significant structural homolog (E > 0.05)")

        ax_c.legend(handles=[
            Patch(facecolor=PAL["obelisk"], edgecolor=PAL["obelisk_d"],
                  linewidth=0.5, label="AF3 input"),
            Patch(facecolor=PAL["delta"], edgecolor="#1F3A52",
                  linewidth=0.5, label="Boltz-2 input"),
            Line2D([0], [0], color=PAL["accent_g"], linestyle=(0, (4, 2.4)),
                   linewidth=1.2, label="E = 0.05"),
        ], loc="lower right", bbox_to_anchor=(1.0, 0.02),
           fontsize=FONT_LEGEND - 0.7, handletextpad=0.4,
           labelspacing=0.30, borderaxespad=0.2)

        grid(ax_c, y=False, x=True)
        panel(ax_c, "C", x=-0.18, y=1.06)

    _save(fig, "Figure4_oblin1_structure")

def fig5_v2():
    print("fig5_v2: topology landscape (panels A + B)")
    fig = plt.figure(figsize=(7.20, 3.90))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.05, 0.95],
        wspace=0.34,
        left=0.085, right=0.985, top=0.88, bottom=0.20,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    cat = _safe_load("catalog_features.json")
    sixs = _safe_load("sixs_comparison.json")
    pos = _safe_load("positive_controls.json")

    if not cat:
        _missing_panel(ax_a, "catalog_features.json missing", letter="A")
    else:
        ob_fp = np.array([r["frac_paired"] for r in cat["OBELISK_FULL"]])
        ob_sld = np.array([r["sld_per100nt"]
                           for r in cat["OBELISK_FULL"]])
        sh_fp = np.array([r["frac_paired"] for r in cat.get("SHUFFLED_CONTROL", [])])
        sh_sld = np.array([r["sld_per100nt"]
                           for r in cat.get("SHUFFLED_CONTROL", [])])

        ax_a.fill_between([0.65, 0.95], 10, 20, color=PAL["accent_g"],
                          alpha=0.10, zorder=0,
                          edgecolor=PAL["accent_g"], linewidth=0.6)

        if sh_fp.size:
            try:
                sns.kdeplot(x=sh_fp, y=sh_sld, ax=ax_a, levels=5,
                            color="#7E7E7E", linewidths=0.55,
                            alpha=0.55, bw_adjust=1.1, thresh=0.05,
                            zorder=1)
            except Exception:
                pass
        try:
            sns.kdeplot(x=ob_fp, y=ob_sld, ax=ax_a, levels=6, fill=True,
                        color=PAL["obelisk"], alpha=0.55, bw_adjust=1.0,
                        thresh=0.05, zorder=2)
            sns.kdeplot(x=ob_fp, y=ob_sld, ax=ax_a, levels=6,
                        color=PAL["obelisk_d"], linewidths=0.55,
                        bw_adjust=1.0, thresh=0.05, zorder=3)
        except Exception:
            ax_a.scatter(ob_fp, ob_sld, s=8, color=PAL["obelisk"], alpha=0.30)

        if pos:
            delta = pos["delta_class"]["rows"]
            if delta:
                ax_a.scatter([r["frac_paired"] for r in delta],
                             [r["sld_per100nt"] for r in delta],
                             s=28, marker="D", color=PAL["delta"],
                             edgecolor="white", linewidth=0.55, zorder=6)
            vir = [r for r in pos["viroids"]["rows"] if r.get("brrc_pass")]
            if vir:
                ax_a.scatter([r["frac_paired"] for r in vir],
                             [r["sld_per100nt"] for r in vir],
                             s=34, marker="^", color=PAL["accent_p"],
                             edgecolor="white", linewidth=0.55, zorder=6)
        if sixs:
            ax_a.scatter([sixs["frac_paired"]],
                         [sixs["sld_per100nt"]],
                         s=120, marker="*", color=PAL["metal"],
                         edgecolor="black", linewidth=0.5, zorder=10)

        ax_a.axvline(0.65, color="#3A3A3A", linestyle=(0, (4, 2.4)),
                     linewidth=0.85, alpha=0.85)
        ax_a.axhline(10, color="#3A3A3A", linestyle=(0, (4, 2.4)),
                     linewidth=0.85, alpha=0.85)
        ax_a.set_xlim(0.30, 0.92)
        ax_a.set_ylim(2, 18.5)

        ax_a.set_xlabel("Fraction paired")
        ax_a.set_ylabel("Small-loop density per 100 nt")
        title_left(ax_a, "BRRC topology landscape (2 of 3 criteria shown)")

        ax_a.legend(handles=[
            Patch(facecolor=PAL["obelisk"], alpha=0.55,
                  edgecolor=PAL["obelisk_d"], linewidth=0.5,
                  label=f"Obelisks (n={len(ob_fp):,})"),
            Patch(facecolor="#CFCFCF", alpha=0.6, edgecolor="#7E7E7E",
                  linewidth=0.5,
                  label=f"Dinuc. shuffles (n={len(sh_fp):,})"),
            Line2D([0], [0], marker="D", color="none",
                   markerfacecolor=PAL["delta"], markeredgecolor="white",
                   markersize=4.8, label="HDV / delta"),
            Line2D([0], [0], marker="^", color="none",
                   markerfacecolor=PAL["accent_p"], markeredgecolor="white",
                   markersize=5.2, label="viroids (BRRC pass)"),
            Line2D([0], [0], marker="*", color="none",
                   markerfacecolor=PAL["metal"], markeredgecolor="black",
                   markersize=7.5, label="6S RNA"),
            Patch(facecolor=PAL["accent_g"], alpha=0.20, edgecolor=PAL["accent_g"],
                  linewidth=0.5, label="BRRC pass (fp + sld)"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, columnspacing=1.2,
           borderaxespad=0.0)
        grid(ax_a)
        panel(ax_a, "A", x=-0.14, y=1.06)

    if not cat:
        _missing_panel(ax_b, "catalog_features.json missing", letter="B")
    else:
        ob_L = np.array([r.get("L", r.get("length"))
                         for r in cat["OBELISK_FULL"]])
        ob_fp = np.array([r["frac_paired"] for r in cat["OBELISK_FULL"]])
        try:
            sns.kdeplot(x=np.log10(ob_L), y=ob_fp, ax=ax_b, levels=6,
                        fill=True, color=PAL["obelisk"], alpha=0.55,
                        bw_adjust=1.0, thresh=0.05, zorder=2)
            sns.kdeplot(x=np.log10(ob_L), y=ob_fp, ax=ax_b, levels=6,
                        color=PAL["obelisk_d"], linewidths=0.55,
                        bw_adjust=1.0, thresh=0.05, zorder=3)
        except Exception:
            ax_b.scatter(np.log10(ob_L), ob_fp, s=8,
                         color=PAL["obelisk"], alpha=0.30)
        if pos:
            delta = pos["delta_class"]["rows"]
            if delta:
                ax_b.scatter(np.log10([r["length"] for r in delta]),
                             [r["frac_paired"] for r in delta],
                             s=28, marker="D", color=PAL["delta"],
                             edgecolor="white", linewidth=0.55, zorder=6)
            vir = [r for r in pos["viroids"]["rows"] if r.get("brrc_pass")]
            if vir:
                ax_b.scatter(np.log10([r["length"] for r in vir]),
                             [r["frac_paired"] for r in vir],
                             s=34, marker="^", color=PAL["accent_p"],
                             edgecolor="white", linewidth=0.55, zorder=6)
        if sixs:
            ax_b.scatter([np.log10(sixs["length_nt"])], [sixs["frac_paired"]],
                         s=120, marker="*", color=PAL["metal"],
                         edgecolor="black", linewidth=0.5, zorder=10)

        ax_b.scatter([np.log10(25)], [0.94], s=58, marker="s",
                     color="#1A1A1A", edgecolor="white", linewidth=0.55,
                     zorder=10)

        ax_b.axhline(0.65, color="#3A3A3A", linestyle=(0, (4, 2.4)),
                     linewidth=0.85, alpha=0.85)
        ax_b.set_xlim(np.log10(15), np.log10(4000))
        ax_b.set_ylim(0.30, 1.02)
        ax_b.set_xticks([np.log10(20), np.log10(50), np.log10(100),
                         np.log10(200), np.log10(500),
                         np.log10(1000), np.log10(2000)])
        ax_b.set_xticklabels(["20", "50", "100", "200", "500",
                              "1,000", "2,000"])
        ax_b.set_xlabel("Length (nt, log scale)")
        ax_b.set_ylabel("Fraction paired")
        title_left(ax_b, "Length vs pairing for RNAP-engaged classes")

        ax_b.legend(handles=[
            Patch(facecolor=PAL["obelisk"], alpha=0.55,
                  edgecolor=PAL["obelisk_d"], linewidth=0.5,
                  label="Obelisks"),
            Line2D([0], [0], marker="D", color="none",
                   markerfacecolor=PAL["delta"], markeredgecolor="white",
                   markersize=4.8, label="HDV / delta"),
            Line2D([0], [0], marker="^", color="none",
                   markerfacecolor=PAL["accent_p"], markeredgecolor="white",
                   markersize=5.2, label="viroids (BRRC pass)"),
            Line2D([0], [0], marker="*", color="none",
                   markerfacecolor=PAL["metal"], markeredgecolor="black",
                   markersize=7.5, label="6S RNA"),
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor="#1A1A1A", markeredgecolor="white",
                   markersize=4.6, label="B-form promoter"),
            Line2D([0], [0], color="#3A3A3A",
                   linestyle="--", linewidth=1.0,
                   label="fp = 0.65 (BRRC)"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=3, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.4, columnspacing=1.2,
           borderaxespad=0.0)
        grid(ax_b)
        panel(ax_b, "B", x=-0.18, y=1.06)

    _save(fig, "Figure5_model")

ALL = {1: fig1_v2, 2: fig2_v2, 3: fig3_v2, 4: fig4_v2, 5: fig5_v2}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    which = sorted(ALL) if not argv else [int(a) for a in argv]
    failed: list[int] = []
    for k in which:
        try:
            ALL[k]()
        except Exception as exc:
            traceback.print_exc()
            failed.append(k)
            print(f"  -> fig{k}_v2 FAILED: {exc}")
    print(f"\nFigures saved to {FIG}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)

if __name__ == "__main__":
    main()
LET = ["A", "B", "C", "D", "E", "F"]

supplement_N_GLOBE = (1, 130)
supplement_DOM_A = (130, 175)
supplement_C_HEL = (175, 202)

_supplement_save = _save

def _supplement_load_pdb_ca(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            rn = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            b = float(line[60:66])
            rows.append((rn, x, y, z, b))
    if not rows:
        return {}
    bmax = max(r[4] for r in rows)
    scale = 100.0 if bmax <= 1.5 else 1.0
    return {rn: (x, y, z, b * scale) for rn, x, y, z, b in rows}

def fig1():
    print("supplement fig1: cross-class boxplots")
    m = json.load(open(JSON / "master_features_expanded.json"))
    groups = {
        "Obelisks":   m["OBELISK_n27"],
        "HDV / delta": m["DELTA_n25"],
        "Rfam":        m["RFAM_EXPANDED_n750"],
    }
    cols = {
        "Obelisks":   PAL["obelisk"],
        "HDV / delta": PAL["delta"],
        "Rfam":        PAL["rfam"],
    }
    edges = {
        "Obelisks":   PAL["obelisk_d"],
        "HDV / delta": "#1F3A52",
        "Rfam":        "#4F4F4F",
    }

    metrics = [
        ("mfe_per_nt", "MFE per nt (kcal / nt)",
         [(1, 3, "n.s.")], False),
        ("frac_paired", "Fraction paired",
         [(1, 3, "***"), (2, 3, "***")], False),
        ("sld_per100nt", "Small loops per 100 nt",
         [(1, 3, "***"), (2, 3, "***")], False),
        ("max_loop", "Max loop length (nt)",
         [(1, 3, "***"), (2, 3, "***")], False),
    ]

    fig, axes_grid = plt.subplots(2, 2, figsize=(7.20, 6.60))
    axes = axes_grid.flatten()
    rng = np.random.default_rng(42)

    for i, (ax, (key, label, brackets, _)) in enumerate(zip(axes, metrics)):
        data, ticklabels = [], []
        for g, recs in groups.items():
            v = [r[key] for r in recs]
            data.append(v)
            ticklabels.append(f"{g}\nn={len(v):,}")
        parts = ax.violinplot(data, showmeans=False, showmedians=False,
                              showextrema=False, widths=0.74)
        for pc, g in zip(parts["bodies"], groups):
            pc.set_facecolor(cols[g])
            pc.set_alpha(0.18)
            pc.set_edgecolor("none")
        bp = ax.boxplot(
            data, patch_artist=True, widths=0.34, showfliers=False,
            medianprops={"color": "#1a1a1a", "linewidth": 1.05},
            boxprops={"linewidth": 0.55},
            whiskerprops={"linewidth": 0.55, "color": "#333333"},
            capprops={"linewidth": 0.55, "color": "#333333"})
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(cols[g])
            patch.set_alpha(0.72)
            patch.set_edgecolor(edges[g])
        for k, v in enumerate(data):
            if len(v) > 200:
                sample = list(rng.choice(v, 120, replace=False))
                pt_alpha = 0.18
            else:
                sample = v
                pt_alpha = 0.40
            x = rng.normal(k + 1, 0.05, size=len(sample))
            ax.scatter(x, sample, color="#1A1A1A", alpha=pt_alpha,
                       s=2.0, zorder=10, edgecolor="none")
        ax.set_xticks(range(1, len(ticklabels) + 1))
        ax.set_xticklabels(ticklabels, fontsize=FONT_TICK - 0.3)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", pad=1.4)
        grid(ax)
        title_left(ax, label)
        panel(ax, LET[i], x=-0.16, y=1.06)
        if brackets:
            ymax = max(np.max(d) for d in data)
            ymin = min(np.min(d) for d in data)
            span = ymax - ymin
            head = 0.22 if len(brackets) == 1 else 0.36
            ax.set_ylim(ymin - span * 0.06, ymax + span * head)
            for j, (a, b, txt) in enumerate(brackets):
                sig_bar(ax, a, b, ymax + span * (0.08 + 0.14 * j),
                        txt, h=span * 0.022, lw=0.85,
                        fs=FONT_ANNOT - 0.4)

    legend_handles = [
        Patch(facecolor=PAL["obelisk"], edgecolor=PAL["obelisk_d"],
              linewidth=0.5, label="Obelisks (n = 27)"),
        Patch(facecolor=PAL["delta"], edgecolor="#1F3A52",
              linewidth=0.5, label="HDV / delta (n = 25)"),
        Patch(facecolor=PAL["rfam"], edgecolor="#4F4F4F",
              linewidth=0.5, label="Rfam panel (n = 750)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.005), frameon=False,
               fontsize=FONT_LEGEND, handletextpad=0.5,
               columnspacing=1.4, labelspacing=0.30)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.075,
                        wspace=0.32, hspace=0.55)
    _supplement_save(fig, "FigureS1_cross_class")

def fig4():
    print("supplement fig4: Oblin-1 composition histograms")
    rows = json.load(open(JSON / "oblin1_conservation.json"))

    panels = [
        ("pos_frac",     "Basic fraction (K + R)",     30),
        ("net_charge",   "Net charge",                 30),
        ("hydrophobicity","Mean hydrophobicity (KD)",  30),
        ("disorder",     "Disorder score (Uversky)",   30),
        ("aromatic",     "Aromatic fraction (F+W+Y)",  30),
        ("has_metal_window6",
         "Has D/E/H/C cluster (6-aa window)",          2),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.20, 4.80))
    for i, (ax, (key, label, bins)) in enumerate(zip(axes.flatten(), panels)):
        v = np.array([r[key] for r in rows], dtype=float)
        n_unique = len(np.unique(v))
        is_binary = (n_unique == 2 and set(np.unique(v)).issubset({0.0, 1.0}))

        if is_binary:
            n_yes = int(v.sum())
            n_no = int(len(v) - n_yes)
            ax.bar([0, 1], [n_no, n_yes],
                   color=[PAL["shuffle"], PAL["delta"]],
                   edgecolor=["#7E7E7E", "#1F3A52"],
                   linewidth=0.55, width=0.55, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No", "Yes"], fontsize=FONT_TICK)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylabel(f"# Oblin-1 proteins (n = {len(v):,})")
            for xi, ni in zip([0, 1], [n_no, n_yes]):
                ax.text(xi, ni + max(n_no, n_yes) * 0.025,
                        f"{ni:,}\n({100 * ni / len(v):.1f}%)",
                        ha="center", va="bottom",
                        fontsize=FONT_TICK - 0.4,
                        linespacing=1.15, fontweight="bold")
            ax.set_ylim(0, max(n_no, n_yes) * 1.30)
            ax.set_xlabel(label)
            grid(ax)
            title_left(ax, label)
            panel(ax, LET[i], x=-0.20, y=1.06)
            continue

        ax.hist(v, bins=bins, color=PAL["obelisk"], alpha=0.70,
                edgecolor="white", linewidth=0.30, density=True, zorder=3)
        is_integer_narrow = np.allclose(v, np.round(v)) and n_unique <= 12
        if n_unique > 5 and not is_integer_narrow:
            xs = np.linspace(v.min(), v.max(), 300)
            try:
                ax.plot(xs, gaussian_kde(v)(xs),
                        color=PAL["obelisk_d"], linewidth=0.95, zorder=4)
            except (np.linalg.LinAlgError, ValueError):
                pass
        mn = float(np.mean(v))
        md = float(np.median(v))
        xspan = float(v.max() - v.min()) if v.size else 1.0
        same_label = f"{mn:.2f}" == f"{md:.2f}"
        same_position = xspan > 0 and abs(mn - md) / xspan < 0.03
        data_ymax = ax.get_ylim()[1]
        if same_label and same_position:
            ax.plot([mn, mn], [0, data_ymax], color="#1A1A1A",
                    linestyle=(0, (4, 2.4)), linewidth=0.85, zorder=5,
                    label=f"mean = median = {mn:.2f}")
        else:
            ax.plot([mn, mn], [0, data_ymax], color="#1A1A1A",
                    linestyle=(0, (4, 2.4)), linewidth=0.85, zorder=5,
                    label=f"mean = {mn:.2f}")
            ax.plot([md, md], [0, data_ymax], color=PAL["delta"],
                    linestyle=":", linewidth=0.85, zorder=5,
                    label=f"median = {md:.2f}")
        if v.size:
            pad = max(xspan * 0.04, 1e-6)
            ax.set_xlim(v.min() - pad, v.max() + pad)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_ylim(0, data_ymax * 1.30)
        leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
                        fontsize=FONT_LEGEND - 0.7,
                        handletextpad=0.4, labelspacing=0.30,
                        borderaxespad=0.4, frameon=True,
                        framealpha=1.0)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#CCCCCC")
        leg.get_frame().set_linewidth(0.5)
        leg.set_zorder(20)
        grid(ax)
        title_left(ax, label)
        panel(ax, LET[i], x=-0.20, y=1.06)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.10,
                        hspace=0.55, wspace=0.34)
    _supplement_save(fig, "FigureS2_oblin1_composition")

def _read_a3m(path):
    seqs, ids = [], []
    cur_id, cur = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id and cur:
                    seqs.append("".join(cur))
                    ids.append(cur_id)
                cur_id = line[1:].split()[0]
                cur = []
            elif line:
                cur.append(line)
        if cur_id:
            seqs.append("".join(cur))
            ids.append(cur_id)
    return seqs, ids

def fig8():
    print("supplement fig8: environmental MSA conservation + APC-MI")
    a3m = PROC / "oblin1_colabfold" / "bfd.mgnify30.metaeuk30.smag30.a3m"
    seqs_a, _ = _read_a3m(a3m)
    query = seqs_a[0]
    L = len(query)
    keep = lambda s: "".join(c for c in s if c.isupper() or c == "-")
    aligned = [keep(s) for s in seqs_a if len(keep(s)) == L]
    cons = []
    for col in range(L):
        chars = [a[col] for a in aligned if a[col] != "-"]
        if not chars:
            cons.append(0)
            continue
        c = Counter(chars)
        cons.append(100 * c.most_common(1)[0][1] / len(chars))
    cons = np.array(cons)

    pairs = json.load(open(JSON / "oblin1_msa_analysis.json"))["top_coevolving_pairs"][:30]

    fig = plt.figure(figsize=(7.20, 6.40))
    gs = fig.add_gridspec(
        2, 2, hspace=0.55, wspace=0.34,
        height_ratios=[1.0, 1.10],
        left=0.085, right=0.985, top=0.92, bottom=0.085,
    )

    ax = fig.add_subplot(gs[0, :])
    pos = np.arange(1, L + 1)
    bcols = [PAL["obelisk"] if c >= 80 else "#C5C5C5" for c in cons]
    ax.bar(pos, cons, color=bcols, edgecolor="none", width=1.0, alpha=0.85)
    smooth = np.convolve(cons, np.ones(10) / 10, mode="same")
    ax.plot(pos, smooth, color="#1A1A1A", linewidth=1.20, alpha=0.95)
    ax.axhline(80, color=PAL["delta"], linestyle=(0, (4, 2.4)),
               linewidth=0.95, alpha=0.85)
    ax.axvspan(*supplement_N_GLOBE, alpha=0.55, color=PAL["bg_n"], zorder=0)
    ax.axvspan(*supplement_DOM_A,   alpha=0.60, color=PAL["bg_a"], zorder=0)
    ax.axvspan(*supplement_C_HEL,   alpha=0.55, color=PAL["bg_c"], zorder=0)
    for p in (154, 155, 156, 158, 160):
        ax.plot([p, p], [0, 100], color=PAL["metal"], linestyle=":",
                linewidth=0.75, alpha=0.85, zorder=2)
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Conservation (% top residue)")
    ax.set_ylim(0, 128)
    ax.set_xlim(0, L + 5)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    for ((a0, a1), lbl, col) in (
            (supplement_N_GLOBE, "N-globule",  "#8E7C3E"),
            (supplement_DOM_A,   "Domain-A",   "#3D5E83"),
            (supplement_C_HEL,   "C-helix",    "#3F6B47")):
        ax.text(0.5 * (a0 + a1), 122, lbl,
                ha="center", va="center",
                fontsize=FONT_TICK, fontweight="semibold",
                color=col, path_effects=text_halo(2.2))

    legend_a = [
        Patch(facecolor=PAL["obelisk"], label="Conservation ≥ 80%"),
        Patch(facecolor="#C5C5C5", label="Conservation < 80%"),
        Line2D([0], [0], color="#1A1A1A", linewidth=1.2,
               label="10-residue smoothed"),
        Line2D([0], [0], color=PAL["delta"], linestyle=(0, (4, 2.4)),
               linewidth=1.2, label="80% threshold"),
        Line2D([0], [0], color=PAL["metal"], linestyle=":",
               linewidth=1.2, label="Domain-A metal residue (D/E/H)"),
    ]
    ax.legend(handles=legend_a, fontsize=FONT_LEGEND - 0.7,
              loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=5, frameon=False,
              handletextpad=0.45, columnspacing=1.20,
              borderaxespad=0.0)
    grid(ax)
    title_left(ax, f"Per-position conservation across "
                   f"{len(aligned)} environmental homologs")
    panel(ax, "A", x=-0.04, y=1.06)

    ax = fig.add_subplot(gs[1, 0])
    xs_raw = np.array([p["i"] for p in pairs])
    ys_raw = np.array([p["j"] for p in pairs])
    mis = np.array([p["APC_MI"] for p in pairs])
    xs_p = np.minimum(xs_raw, ys_raw)
    ys_p = np.maximum(xs_raw, ys_raw)
    sc = ax.scatter(xs_p, ys_p, s=70 * mis, c=mis, cmap="Reds",
                    alpha=0.80, edgecolor="#444444", linewidth=0.40)
    ax.plot([0, L], [0, L], color="#666666", linewidth=0.5, linestyle=":")
    dom_box = Rectangle(
        (supplement_DOM_A[0], supplement_DOM_A[0]),
        supplement_DOM_A[1] - supplement_DOM_A[0], supplement_DOM_A[1] - supplement_DOM_A[0],
        facecolor=PAL["bg_a"], edgecolor=PAL["delta"],
        linewidth=0.55, alpha=0.30, zorder=0,
    )
    ax.add_patch(dom_box)
    pad = max(8, L * 0.05)
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-pad, L + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Residue i (smaller index)")
    ax.set_ylabel("Residue j (larger index)")
    title_left(ax, f"Top {len(pairs)} APC-MI residue pairs")
    cbar = plt.colorbar(sc, ax=ax, label="APC-MI", shrink=0.78)
    cbar.ax.tick_params(labelsize=FONT_TICK - 0.4,
                        width=0.45, length=2.0, pad=1.6)
    cbar.set_label("APC-MI", fontsize=FONT_AXLABEL, labelpad=3)
    ax.legend(handles=[
        Patch(facecolor=PAL["bg_a"], edgecolor=PAL["delta"],
              linewidth=0.55, alpha=0.30,
              label="Domain-A self-block"),
        Line2D([0], [0], color="#666666", linewidth=0.7,
               linestyle=":", label=r"$i = j$ diagonal"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.22),
       ncol=2, fontsize=FONT_LEGEND - 0.7, handletextpad=0.4,
       columnspacing=1.2, borderaxespad=0.0, frameon=False)
    grid(ax, y=False)
    panel(ax, "B", x=-0.18, y=1.06)

    ax = fig.add_subplot(gs[1, 1])
    ca = _supplement_load_pdb_ca(PROC / "oblin1_af3.pdb")
    distances = []
    for p in pairs:
        if p["i"] in ca and p["j"] in ca:
            d = math.sqrt(sum((ca[p["i"]][a] - ca[p["j"]][a]) ** 2
                              for a in range(3)))
            distances.append(d)
    d_max = max(distances) if distances else 1.0
    bin_edges = np.arange(0, math.ceil(d_max / 5.0) * 5.0 + 5.0, 5.0)
    ax.hist(distances, bins=bin_edges, color="#B5B5B5",
            edgecolor="#444444", linewidth=0.40, zorder=3)
    ax.axvline(8, color=PAL["obelisk"], linestyle=(0, (4, 2.4)),
               linewidth=1.10)
    ax.text(8 + 1.0, ax.get_ylim()[1] * 0.92,
            "8 Å contact threshold",
            ha="left", va="top",
            fontsize=FONT_TICK - 0.3, color=PAL["obelisk"],
            fontweight="bold",
            path_effects=text_halo(2.0))
    ax.set_xlabel(r"C$_\alpha$ – C$_\alpha$ distance (Å, AlphaFold3)")
    ax.set_ylabel(f"# of top-{len(distances)} pairs")
    ax.set_xlim(0, bin_edges[-1])
    title_left(ax, "AF3 distance of top APC-MI pairs")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.18)
    grid(ax)
    panel(ax, "C", x=-0.18, y=1.06)

    _supplement_save(fig, "FigureS3_msa_coevolution")

def fig9():
    print("supplement fig9: sigma70 scan + 6S RNA topology")
    prom = json.load(open(JSON / "promoter_scan.json"))
    sixs = json.load(open(JSON / "sixs_comparison.json"))
    fc = json.load(open(JSON / "catalog_features.json"))
    ob = fc["OBELISK_FULL"]
    ob_fp = np.array([r["frac_paired"] for r in ob])
    ob_sld = np.array([r["sld_per100nt"] for r in ob])
    ob_ml = np.array([r["max_loop"] for r in ob])

    fig, axes_grid = plt.subplots(2, 2, figsize=(7.20, 6.20))
    axes = axes_grid.flatten()

    ax = axes[0]
    ob_hist = prom["histogram_obelisk_pair_counts"]
    sh_hist = prom["histogram_shuffle_pair_counts"]
    ks = sorted(int(k) for k in ob_hist.keys())
    obh = [ob_hist[str(k)] for k in ks]
    shh = [sh_hist[str(k)] for k in ks]
    w = 0.40
    xp = np.arange(len(ks))
    ax.bar(xp - w / 2, obh, w, color=PAL["obelisk"],
           edgecolor=PAL["obelisk_d"], linewidth=0.55,
           label=f"Obelisks (n = {prom['n_obelisks']:,})", zorder=3)
    ax.bar(xp + w / 2, shh, w, color=PAL["shuffle"],
           edgecolor="#7E7E7E", linewidth=0.55,
           label=f"Shuffles (n = {prom['n_shuffles']:,})", zorder=3)
    xt = [str(k) if k < 8 else "8+" for k in ks]
    ax.set_xticks(xp)
    ax.set_xticklabels(xt, fontsize=FONT_TICK)
    ax.set_xlabel(r"# canonical $\sigma^{70}$ promoter pairs per sequence")
    ax.set_ylabel("# sequences (log scale)")
    title_left(ax, r"$\sigma^{70}$ promoter pair count "
                   r"(obelisks 0.89× shuffles)")
    ax.set_yscale("log")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(max(0.5, y0), y1 * 2.4)
    ax.legend(loc="upper right", fontsize=FONT_LEGEND - 0.7,
              handletextpad=0.45, labelspacing=0.30,
              borderaxespad=0.4)
    grid(ax)
    panel(ax, "A", x=-0.18, y=1.06)

    titles = ["Fraction paired", "Small loops per 100 nt",
              "Max loop length (nt)"]
    subtitles = ["pairing", "small-loop density", "max loop length"]
    arrs = [ob_fp, ob_sld, ob_ml]
    sixs_vals = [sixs["frac_paired"],
                 sixs["sld_per100nt"],
                 sixs["max_loop_nt"]]
    sixs_fmts = [".2f", ".2f", ".0f"]
    thresholds = [0.65, 10, 15]
    directions = ["above", "above", "below"]
    letters_local = ["B", "C", "D"]

    for ax, title, sub, v, sixs_v, sixs_fmt, thr, direction, letter in zip(
            axes[1:], titles, subtitles, arrs, sixs_vals, sixs_fmts,
            thresholds, directions, letters_local):
        ax.hist(v, bins=44, color=PAL["obelisk"], alpha=0.68,
                edgecolor="white", linewidth=0.30, zorder=3)
        op = ">" if direction == "above" else r"\leq"
        ax.axvline(thr, color="#1A1A1A",
                   linestyle=(0, (4, 2.0)), linewidth=1.0, zorder=4)
        ax.axvline(sixs_v, color=PAL["delta"], linewidth=1.20,
                   alpha=0.95, zorder=4)
        ax.set_xlabel(title)
        ax.set_ylabel("# obelisks")
        title_left(ax, f"6S RNA vs catalog: {sub}")

        if sixs_fmt == ".0f":
            margin_left, margin_right = 4.0, 6.5
        elif sixs_v < 1:
            margin_left = margin_right = 0.028
        else:
            margin_left = margin_right = 0.9
        x_left = min(v.min(), sixs_v) - margin_left
        x_right = max(v.max(), sixs_v) + margin_right
        ax.set_xlim(x_left, x_right)

        y0, y1 = ax.get_ylim()
        if letter == "D":
            data_max = float(np.histogram(v, bins=44)[0].max())
            ax.set_ylim(y0, data_max * 1.34)
        else:
            ax.set_ylim(y0, y1 * 1.45)

        sixs_left = sixs_v < thr
        legend_loc = "upper right" if sixs_left else "upper left"
        brrc_value_txt = (f"{thr:.2f}" if isinstance(thr, float) and thr < 1
                          else f"{thr}")
        ax.legend(handles=[
            Patch(facecolor=PAL["obelisk"], edgecolor="white",
                  label="Obelisks"),
            Line2D([0], [0], color="#1A1A1A",
                   linestyle=(0, (4, 2.0)), linewidth=1.2,
                   label=rf"BRRC ${op}\,{brrc_value_txt}$"),
            Line2D([0], [0], color=PAL["delta"], linewidth=1.4,
                   label=f"6S RNA = {sixs_v:{sixs_fmt}}"),
        ], loc=legend_loc, fontsize=FONT_LEGEND - 0.7,
           handletextpad=0.45, labelspacing=0.30, borderaxespad=0.4)
        grid(ax)
        panel(ax, letter, x=-0.13, y=1.06)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.085,
                        wspace=0.32, hspace=0.55)
    _supplement_save(fig, "FigureS4_promoter_and_6S")

def fig10():
    print("supplement fig10: supporting analyses")
    phylo = json.load(open(JSON / "phylo_oblin1.json"))
    codon = json.load(open(JSON / "codon_usage.json"))
    cmp = json.load(open(JSON / "structure_compare.json"))
    pos = json.load(open(JSON / "positive_controls.json"))
    bac = json.load(open(JSON / "bacterial_controls.json"))

    fig, axes2d = plt.subplots(2, 2, figsize=(7.20, 6.20))
    axes = axes2d.flatten()

    ax = axes[0]
    obelisk_pct = pos["comparator_summary"][
        "ob_circ_pass_pct"]
    n_obelisks = 5169
    n_delta = pos["delta_class"].get("n_attempted", 47)
    n_srp = bac["per_family"].get("RF00177", {}).get("n", 266)
    n_group1 = bac["per_family"].get("RF00028", {}).get("n", 28)
    n_rnasep = bac["per_family"].get("RF00010", {}).get("n", 2)
    n_shuffle = 5169
    classes = [
        ("HDV / delta",   pos["delta_class"]["pass_pct"],     PAL["delta"],   "#1F3A52", n_delta),
        ("Obelisks",      obelisk_pct,                        PAL["obelisk"], PAL["obelisk_d"], n_obelisks),
        ("Bact. SRP RNA", bac["per_family"].get("RF00177", {}).get("pass_pct", 0),
         PAL["accent_g"], "#28523A", n_srp),
        ("Group-I introns", bac["per_family"].get("RF00028", {}).get("pass_pct", 0),
         "#A0A0A0", "#666666", n_group1),
        ("RNase P",       bac["per_family"].get("RF00010", {}).get("pass_pct", 0),
         "#A0A0A0", "#666666", n_rnasep),
        ("Shuffle",       0.30,                               PAL["shuffle"], "#7E7E7E", n_shuffle),
    ]
    classes.sort(key=lambda c: c[1])
    labels = [f"{c[0]}\n(n = {c[4]:,})" for c in classes]
    pcts = [c[1] for c in classes]
    cols = [c[2] for c in classes]
    edges = [c[3] for c in classes]
    y = np.arange(len(labels))
    bars = ax.barh(y, pcts, color=cols, edgecolor=edges, linewidth=0.55,
                   height=0.66, zorder=3)
    for b, v, col in zip(bars, pcts, cols):
        y_center = b.get_y() + b.get_height() / 2
        if v < 1.0:
            ax.plot([0.8, 3.2], [y_center, y_center],
                    color=col, lw=2.6, solid_capstyle="butt",
                    zorder=5, clip_on=False)
            label_x = 4.6
        else:
            label_x = v + 1.8
        ax.text(label_x, y_center,
                f"{v:.1f}%", va="center", ha="left",
                fontsize=FONT_TICK, color="#0F0F0F", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_TICK - 0.3)
    ax.set_xlim(0, 122)
    ax.set_xlabel("BRRC pass rate (%)")
    title_left(ax, "Pass rate by RNA class")
    grid(ax, y=False, x=True)
    panel(ax, "A", x=-0.32, y=1.06)

    ax = axes[1]
    obs = phylo["obs_clade_var"]
    null_samples = phylo.get("perm_null_var_samples")
    if null_samples:
        null_arr = np.asarray(null_samples, dtype=float)
        ax.hist(null_arr, bins=24, color=PAL["rfam"], edgecolor="white",
                linewidth=0.30, zorder=3,
                label=f"Label-permutation null (n={len(null_arr)})")
        ax.axvline(obs, color=PAL["accent_p"], linewidth=1.30,
                   ymin=0.0, ymax=0.80,
                   label=f"Observed = {obs:.0f}")
        ax.set_xlim(min(null_arr.min(), obs) * 0.92,
                    max(null_arr.max(), obs) * 1.10)
        n_ge = int(np.sum(null_arr >= obs))
        p_text = r"$p < 0.005$" if n_ge == 0\
            else rf"$p = {n_ge / len(null_arr):.3f}$"
        y0_now, y1_now = ax.get_ylim()
        ax.text(obs * 0.985, y1_now * 0.85, p_text,
                ha="right", va="center",
                fontsize=FONT_ANNOT, color=PAL["accent_p"],
                fontweight="bold",
                path_effects=text_halo(2.4))
        legend_loc = "upper left"
    else:
        null_mean = phylo["perm_null_var_mean"]
        null_std = phylo.get("perm_null_var_std", 0.0)
        ax.errorbar([0], [null_mean], yerr=[[null_std], [null_std]],
                    fmt="s", color=PAL["rfam"], capsize=6,
                    label="null mean ± SD (n=200)")
        ax.scatter([1], [obs], marker="*", s=140, color=PAL["accent_p"],
                   zorder=5, label=f"Observed = {obs:.0f}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["null", "observed"])
        ax.set_xlim(-0.5, 1.5)
        legend_loc = "upper center"
    ax.set_xlabel("Variance of clade-level BRRC pass rate")
    ax.set_ylabel("Permutation count")
    title_left(ax, "Phylogenetic-permutation null")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.18)
    ax.legend(loc=legend_loc, fontsize=FONT_LEGEND - 0.7,
              handletextpad=0.45, labelspacing=0.30,
              borderaxespad=0.4)
    grid(ax)
    panel(ax, "B", x=-0.18, y=1.06)

    ax = axes[2]
    cos_host = codon["cosine_obelisk_vs_S_sanguinis"]
    cos_eco = codon["cosine_obelisk_vs_E_coli"]
    xs = np.array([0, 1])
    bars = ax.bar(xs, [cos_host, cos_eco],
                  color=[PAL["delta"], PAL["rfam"]], width=0.55,
                  edgecolor=["#1F3A52", "#4F4F4F"], linewidth=0.55,
                  zorder=3)
    for b, v in zip(bars, [cos_host, cos_eco]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.020,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=FONT_ANNOT, color="#0F0F0F", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([r"$S.\,sanguinis$" "\n" r"(host)",
                        r"$E.\,coli$" "\n" r"(reference)"],
                       fontsize=FONT_TICK)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Cosine similarity")
    title_left(ax, "Obelisk-1 codon usage")
    grid(ax)
    panel(ax, "C", x=-0.18, y=1.06)

    ax = axes[3]
    rmsd_n = cmp["rmsd_A_N_globule_aligned_locally"]
    rmsd_global = cmp["rmsd_A_after_global_Kabsch"]
    xs = np.array([0, 1])
    bars = ax.bar(xs, [rmsd_n, rmsd_global],
                  color=[PAL["obelisk"], PAL["rfam"]], width=0.55,
                  edgecolor=[PAL["obelisk_d"], "#4F4F4F"], linewidth=0.55,
                  zorder=3)
    y_pad = max(rmsd_n, rmsd_global) * 0.025
    for b, v in zip(bars, [rmsd_n, rmsd_global]):
        ax.text(b.get_x() + b.get_width() / 2, v + y_pad,
                f"{v:.2f} Å", ha="center", va="bottom",
                fontsize=FONT_ANNOT, color="#0F0F0F", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["N-globule\n(rigid core)", "Global"],
                       fontsize=FONT_TICK)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, max(rmsd_n, rmsd_global) * 1.20)
    ax.set_ylabel(r"C$_\alpha$ RMSD (Å, Kabsch)")
    title_left(ax, "AF3 vs Boltz-2 RMSD")
    grid(ax)
    panel(ax, "D", x=-0.18, y=1.06)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.085,
                        wspace=0.36, hspace=0.55)
    _supplement_save(fig, "FigureS5_supporting_analyses")

supplement_ALL = {1: fig1, 4: fig4, 8: fig8, 9: fig9, 10: fig10}

def supplement_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        which = sorted(supplement_ALL.keys())
    else:
        which = [int(a) for a in argv]
    for k in which:
        supplement_ALL[k]()
    print(f"\nFigures saved to {FIG}")

if __name__ == "__main__":
    supplement_main()

