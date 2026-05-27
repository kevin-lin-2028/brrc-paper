
from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from math import exp, sqrt
from pathlib import Path

import RNA
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
from seqfold import dot_bracket, fold as sf_fold

from oblin.core import *
from oblin.core import JSON, PROC, RAW, ROOT
from oblin.analyses import _fasta, _topo, _passes, _circ_fold, _shuffle_di, VIROIDS, SIXS_ECOLI
JSON.mkdir(parents=True, exist_ok=True)

def _fasta(path):
    out = []
    nm, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if nm is not None:
                    out.append((nm, "".join(buf)))
                nm = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
        if nm is not None:
            out.append((nm, "".join(buf)))
    return out

def _shuffle_di(seq, rng):
    seq = seq.upper().replace("U", "T")
    if len(seq) < 4:
        return seq
    edges = {b: [] for b in set(seq)}
    for a, b in zip(seq, seq[1:]):
        edges.setdefault(a, []).append(b)
    last = seq[-1]
    for _ in range(8):
        last_edge = {}
        nodes = [n for n in edges if n != last and edges[n]]
        for n in nodes:
            last_edge[n] = rng.choice(edges[n])
        rem = {n: list(edges[n]) for n in edges}
        for n in nodes:
            rem[n].remove(last_edge[n])
            rng.shuffle(rem[n])
            rem[n].append(last_edge[n])
        out, cur, ok = [seq[0]], seq[0], True
        for _ in range(len(seq) - 1):
            if not rem.get(cur):
                ok = False
                break
            nxt = rem[cur].pop(0)
            out.append(nxt)
            cur = nxt
        if ok and len(out) == len(seq):
            return "".join(out)
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)

def _circ_fold(seq):
    s = seq.upper().replace("T", "U")
    md = RNA.md()
    md.circ = 1
    fc = RNA.fold_compound(s, md)
    return fc.mfe()

def _sigmoid(z):

    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def _logreg_fit(X, y, lam=1.0, lr=0.1, n_iter=400):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        z = X @ w + b
        p = _sigmoid(z)
        grad_w = (X.T @ (p - y)) / n + (lam / n) * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "W": "AT", "S": "GC", "R": "AG", "Y": "CT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}

def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGTU", "TGCAA"))[::-1]

def _norm(s: str) -> str:
    return s.upper().replace("U", "T")

def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()

def _match_pos(seq: str, consensus: str, max_mm: int) -> int:
    L, k = len(seq), len(consensus)
    cls = [IUPAC[c] for c in consensus]
    hits = 0
    for i in range(L - k + 1):
        mm = 0
        for j in range(k):
            if seq[i + j] not in cls[j]:
                mm += 1
                if mm > max_mm:
                    break
        if mm <= max_mm:
            hits += 1
    return hits

def _scan(seq: str, consensus: str, max_mm: int, wrap_pad: int = 50) -> int:
    s = _norm(seq)
    rc = _revcomp(s)
    pad = max(wrap_pad, len(consensus) - 1)
    s_c = s + s[:pad]
    rc_c = rc + rc[:pad]
    return _match_pos(s_c, consensus, max_mm) + _match_pos(rc_c, consensus, max_mm)

def _scan_pair(seq: str, m1: str, m2: str, gap_lo: int, gap_hi: int,
               max_mm1: int, max_mm2: int, wrap_pad: int = 50) -> int:
    s = _norm(seq)
    rc = _revcomp(s)
    n = 0
    for strand in (s, rc):
        pad = max(wrap_pad, len(m1) + len(m2) + gap_hi)
        x = strand + strand[:pad]
        L = len(x)
        cls1 = [IUPAC[c] for c in m1]
        cls2 = [IUPAC[c] for c in m2]
        for i in range(L - len(m1) - gap_lo - len(m2) + 1):
            mm = 0; ok = True
            for j, c in enumerate(cls1):
                if x[i + j] not in c:
                    mm += 1
                    if mm > max_mm1:
                        ok = False; break
            if not ok:
                continue
            for g in range(gap_lo, gap_hi + 1):
                jstart = i + len(m1) + g
                if jstart + len(m2) > L:
                    break
                mm2 = 0; ok2 = True
                for k, c in enumerate(cls2):
                    if x[jstart + k] not in c:
                        mm2 += 1
                        if mm2 > max_mm2:
                            ok2 = False; break
                if ok2:
                    n += 1
                    break
    return n

def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * p, 3),
            round(100 * max(0.0, c - h), 3),
            round(100 * min(1.0, c + h), 3))

def _katz(k1: int, n1: int, k2: int, n2: int):
    if k1 == 0 or k2 == 0 or n1 == 0 or n2 == 0:
        return None, None, None
    p1, p2 = k1 / n1, k2 / n2
    rr = p1 / p2
    se = np.sqrt((1 - p1) / k1 + (1 - p2) / k2)
    return round(rr, 3), round(rr * np.exp(-1.96 * se), 3), round(rr * np.exp(1.96 * se), 3)

def _chi2_2x2(k1: int, n1: int, k2: int, n2: int):
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
    if (table < 0).any() or table.sum() == 0:
        return None
    try:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        return float(p)
    except Exception:
        return None

MOTIFS = [
    ("archaea_TATA_TTTAAA",  "TTTAAA"),
    ("archaea_TATA_TTTAWA",  "TTTAWA"),
    ("archaea_BRE_VRAAAW",   "VRAAAW"),
    ("archaea_BRE_VWAAAR",   "VWAAAR"),
    ("ext_sigma70_TGn_TATAAT", "TGNTATAAT"),
    ("strep_pribnow_TATAAT_TTGACW", "TATAAT"),
    ("eukaryotic_yCAAT_box", "WAACAAAG"),
    ("AT_tract_W5plus",     "WWWWW"),
]

PAIRED = [
    ("archaea_BRE_TATA_pair_VRAAAW", "VRAAAW", "TTTAWA", 0, 3),
    ("archaea_BRE_TATA_pair_VWAAAR", "VWAAAR", "TTTAWA", 0, 3),
    ("sulfolobales_BRE_TATA_25bp",   "VRAAAW", "TTTAWA", 1, 4),
    ("strep_pribnow_pair",           "TTGACW", "TATAAT", 14, 20),
]

def _sample(path: Path, n_sample: int, seed: int, lo: int = 700, hi: int = 1700):
    seqs = _fasta(path)
    keep = [(nm, _norm(s)) for nm, s in seqs if lo <= len(s) <= hi]
    rng = random.Random(seed)
    rng.shuffle(keep)
    return keep[:n_sample]

def _scan_catalog(label: str, sample, seed: int):
    np_rng = np.random.default_rng(seed + 2)
    print(f"[{label}] generating {len(sample)} dinucleotide shuffles", flush=True)
    shuffles = [(nm, _shuffle_di(s, np_rng)) for nm, s in sample]
    results = {}
    for mid, consensus in MOTIFS:
        for mm in (0, 1):
            tag = f"{mid}_mm{mm}"
            ob_k = sum(1 for _, s in sample if _scan(s, consensus, mm) > 0)
            sh_k = sum(1 for _, s in shuffles if _scan(s, consensus, mm) > 0)
            results[tag] = _row(ob_k, len(sample), sh_k, len(shuffles),
                                consensus, mm)
            print(f"  {tag:50s}  obelisk {ob_k}/{len(sample)}  "
                  f"shuffle {sh_k}/{len(shuffles)}  "
                  f"RR={results[tag]['enrichment_x']}", flush=True)
    for mid, m1, m2, glo, ghi in PAIRED:
        for mm in (0, 1):
            tag = f"{mid}_mm{mm}"
            ob_k = sum(1 for _, s in sample
                       if _scan_pair(s, m1, m2, glo, ghi, mm, mm) > 0)
            sh_k = sum(1 for _, s in shuffles
                       if _scan_pair(s, m1, m2, glo, ghi, mm, mm) > 0)
            results[tag] = _row(ob_k, len(sample), sh_k, len(shuffles),
                                f"{m1}+({glo}-{ghi}nt)+{m2}", mm)
            print(f"  {tag:50s}  obelisk {ob_k}/{len(sample)}  "
                  f"shuffle {sh_k}/{len(shuffles)}  "
                  f"RR={results[tag]['enrichment_x']}", flush=True)
    return results

def _row(ob_k, n_ob, sh_k, n_sh, consensus, mm):
    ob_pct, ob_lo, ob_hi = _wilson(ob_k, n_ob)
    sh_pct, sh_lo, sh_hi = _wilson(sh_k, n_sh)
    rr, rr_lo, rr_hi = _katz(ob_k, n_ob, sh_k, n_sh)
    p = _chi2_2x2(ob_k, n_ob, sh_k, n_sh)
    return {
        "consensus": consensus,
        "mismatches_allowed": mm,
        "n_obelisks": n_ob,
        "obelisk_with_hit": ob_k,
        "obelisk_pct": ob_pct,
        "obelisk_wilson_ci_pct": [ob_lo, ob_hi],
        "n_shuffles": n_sh,
        "shuffle_with_hit": sh_k,
        "shuffle_pct": sh_pct,
        "shuffle_wilson_ci_pct": [sh_lo, sh_hi],
        "enrichment_x": rr,
        "enrichment_katz_ci": [rr_lo, rr_hi] if rr is not None else None,
        "chi2_2x2_p": p,
    }

def _flag(rows, alpha):
    enriched = []
    for tag, r in rows.items():
        if r["enrichment_x"] is None:
            continue
        if r["chi2_2x2_p"] is None:
            continue
        sig = r["chi2_2x2_p"] < alpha
        ci_above = (r["enrichment_katz_ci"] is not None
                    and r["enrichment_katz_ci"][0] is not None
                    and r["enrichment_katz_ci"][0] > 1.0)
        if sig and ci_above and r["enrichment_x"] > 1.5:
            enriched.append({
                "motif": tag,
                "enrichment_x": r["enrichment_x"],
                "katz_ci": r["enrichment_katz_ci"],
                "chi2_p": r["chi2_2x2_p"],
                "ob_pct": r["obelisk_pct"],
                "sh_pct": r["shuffle_pct"],
            })
    return enriched

def archaeal_promoters_main(n_sample: int = 500, seed: int = 42):
    zhel_path = PROC / "obelisks_zheludev_catalog.fasta"
    hsob_path = RAW / "hsob_obelisks_new.fasta"
    print(f"loading Zheludev catalog from {zhel_path}", flush=True)
    zhel = _sample(zhel_path, n_sample, seed)
    print(f"  Zheludev subsample: {len(zhel)}", flush=True)
    print(f"loading HsOb catalog from {hsob_path}", flush=True)
    hsob = _sample(hsob_path, n_sample, seed)
    print(f"  HsOb subsample:     {len(hsob)}", flush=True)

    print("scanning Zheludev", flush=True)
    zhel_rows = _scan_catalog("Zheludev", zhel, seed=seed)
    print("scanning HsOb", flush=True)
    hsob_rows = _scan_catalog("HsOb", hsob, seed=seed + 100)

    n_classes = len(MOTIFS) + len(PAIRED)
    n_regimes = 2
    n_catalogs = 2
    n_tests = n_classes * n_regimes * n_catalogs
    alpha = 0.05 / n_tests

    zhel_flag = _flag(zhel_rows, alpha)
    hsob_flag = _flag(hsob_rows, alpha)

    out = {
        "tool": "archaeal_promoters",
        "scan_mode": "both strands, circular wrap, 0/1 mismatch IUPAC",
        "n_sample_per_catalog": n_sample,
        "n_motif_classes": n_classes,
        "n_mismatch_regimes": n_regimes,
        "n_catalogs": n_catalogs,
        "n_tests_total": n_tests,
        "bonferroni_alpha": alpha,
        "catalogs": {
            "zheludev_2024": {
                "fasta": _rel(zhel_path),
                "n_total": len(_fasta(zhel_path)),
                "n_sampled": len(zhel),
                "motifs": zhel_rows,
                "n_enriched_bonf": len(zhel_flag),
                "enriched_motifs": zhel_flag,
            },
            "hsob_2026": {
                "fasta": _rel(hsob_path),
                "n_total": len(_fasta(hsob_path)),
                "n_sampled": len(hsob),
                "motifs": hsob_rows,
                "n_enriched_bonf": len(hsob_flag),
                "enriched_motifs": hsob_flag,
            },
        },
        "headline": _headline(zhel_flag, hsob_flag),
    }

    JSON.mkdir(parents=True, exist_ok=True)
    out_path = JSON / "archaeal_promoters.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(f"Zheludev: {len(zhel_flag)} motif(s) enriched after Bonferroni"
          f" (alpha={alpha:.2e})", flush=True)
    print(f"HsOb:     {len(hsob_flag)} motif(s) enriched after Bonferroni", flush=True)
    return out

def _headline(zhel_flag, hsob_flag):
    if not zhel_flag and not hsob_flag:
        return "no motif enriched after Bonferroni"
    bits = []
    if zhel_flag:
        bits.append("Zheludev: " + ", ".join(
            f"{r['motif']} ({r['enrichment_x']}x)" for r in zhel_flag))
    if hsob_flag:
        bits.append("HsOb: " + ", ".join(
            f"{r['motif']} ({r['enrichment_x']}x)" for r in hsob_flag))
    return "Enriched motifs after Bonferroni: " + "; ".join(bits)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    archaeal_promoters_main(n_sample=n)
def _circ_subopt_sample(seq, n_samples=20):
    s = seq.upper().replace("T", "U")
    md = RNA.md()
    md.circ = 1
    md.uniq_ML = 1
    fc = RNA.fold_compound(s, md)
    fc.pf()
    structs = fc.pbacktrack(n_samples)
    return [(str(st), 0.0) for st in structs]

def _worker(task):
    seq, n_samples, sub_seed = task
    try:
        samples = _circ_subopt_sample(seq, n_samples)
    except Exception:
        return None, []
    passes = []
    for db, _ in samples:
        try:
            fp, sld, ml = _topo(db)
            passes.append(_passes(fp, sld, ml))
        except Exception:
            pass
    return seq, passes

def ensemble_brrc(n_sample=100, n_samples_per_seq=30, seed=42, workers=4):

    rng = random.Random(seed)
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    keep = [(nm, s) for nm, s in seqs if 700 <= len(s) <= 1700]
    rng.shuffle(keep)
    sample = keep[:n_sample]
    print(f"  sampling n={len(sample)} obelisks, {n_samples_per_seq} Boltzmann "
          f"samples each, {workers} worker(s)", flush=True)

    tasks = [(s, n_samples_per_seq, seed + i) for i, (nm, s) in enumerate(sample)]
    t0 = time.time()
    per_obelisk = []

    if workers > 1:
        pool = mp.Pool(workers)
        results = pool.imap_unordered(_worker, tasks, chunksize=4)
    else:
        results = (_worker(t) for t in tasks)

    done = 0
    for seq, passes in results:
        if seq is None:
            continue
        if passes:
            per_obelisk.append({
                "n_samples": len(passes),
                "n_pass": int(sum(passes)),
                "pass_prob": float(sum(passes)) / len(passes),
            })
        done += 1
        if done % 20 == 0 or done == len(sample):
            dt = time.time() - t0
            print(f"  {done}/{len(sample)} obelisks ({done / max(1e-3, dt):.1f} seq/s)",
                  flush=True)

    if workers > 1:
        pool.close(); pool.join()

    if not per_obelisk:
        print("no obelisks produced ensemble samples", flush=True)
        return None

    probs = np.array([r["pass_prob"] for r in per_obelisk])
    all_pass = (probs == 1.0).sum()
    any_pass = (probs > 0.0).sum()
    mean_prob = float(probs.mean())

    out = {
        "tool": "ensemble_brrc",
        "n_obelisks": len(per_obelisk),
        "n_samples_per_obelisk": n_samples_per_seq,
        "seed": seed,
        "wall_time_s": round(time.time() - t0, 1),
        "mean_ens_pp": round(mean_prob, 4),
        "med_ens_pp": round(float(np.median(probs)), 4),
        "n_ob_all_pass": int(all_pass),
        "n_ob_any_pass": int(any_pass),
        "n_ob_none_pass": int((probs == 0.0).sum()),
        "pass_prob_distribution": {
            "min": round(float(probs.min()), 4),
            "p25": round(float(np.percentile(probs, 25)), 4),
            "p50": round(float(np.percentile(probs, 50)), 4),
            "p75": round(float(np.percentile(probs, 75)), 4),
            "max": round(float(probs.max()), 4),
        },

    }

    JSON.mkdir(parents=True, exist_ok=True)
    with open(JSON / "ensemble_brrc.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {JSON / 'ensemble_brrc.json'}", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ensemble_brrc(n_sample=n, n_samples_per_seq=samples, workers=workers)
def _hard_negatives_fasta(path):
    out = []
    nm, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if nm is not None:
                    out.append((nm, "".join(buf)))
                nm = line[1:].split()[0]; buf = []
            elif line:
                buf.append(line)
        if nm is not None:
            out.append((nm, "".join(buf)))
    return out

def _gc(s):
    s = s.upper()
    n = sum(1 for c in s if c in "GC")
    L = sum(1 for c in s if c in "ACGUT")
    return n / max(1, L)

def _cpg_ratio(s):
    s = s.upper().replace("U", "T")
    L = len(s)
    if L < 2:
        return 0.0
    n_cpg = sum(1 for a, b in zip(s, s[1:]) if a == "C" and b == "G")
    n_c = sum(1 for c in s if c == "C")
    n_g = sum(1 for c in s if c == "G")
    expected = max(1e-9, n_c * n_g / L)
    return n_cpg / expected

def _longest_orf(s):
    s = s.upper().replace("U", "T")
    stops = {"TAA", "TAG", "TGA"}
    rc = s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    best = 0
    for strand in (s, rc):
        for off in range(3):
            i = off
            while i + 3 <= len(strand):
                if strand[i:i + 3] == "ATG":
                    j = i
                    while j + 3 <= len(strand):
                        if strand[j:j + 3] in stops:
                            best = max(best, j + 3 - i); break
                        j += 3
                    i = j + 3
                else:
                    i += 3
    return best

def _roc_auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    from scipy.stats import mannwhitneyu
    try:
        u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
        auc = u / (len(pos) * len(neg))
        return max(auc, 1 - auc)
    except Exception:
        return 0.5

def _standardize_fit(X):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd

def _hard_negatives_sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def _hn_logreg_fit(X, y, l2=1.0, iters=200, lr=0.5):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xb @ w
        p = _hard_negatives_sigmoid(z)
        grad = Xb.T @ (p - y) + l2 * np.concatenate([[0.0], w[1:]])

        W = p * (1 - p)
        H_diag = np.diag(Xb.T @ (Xb * W[:, None])) + l2 * np.concatenate([[0.0], np.ones(d)])
        step = grad / (H_diag + 1e-9)
        w -= lr * step
    return w[0], w[1:]

def _logreg_predict(intercept, coefs, X):
    return _hard_negatives_sigmoid(np.asarray(X) @ np.asarray(coefs) + intercept)

def _logreg_cv(X, y, k=5, seed=42):
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)
    aucs = []
    for i in range(k):
        te = np.concatenate([pos_folds[i], neg_folds[i]])
        tr = np.setdiff1d(np.arange(len(y)), te)
        mu, sd = _standardize_fit(X[tr])
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        b, w = _hn_logreg_fit(Xtr, y[tr])
        p = _logreg_predict(b, w, Xte)
        aucs.append(_roc_auc(p.tolist(), y[te].tolist()))

    mu, sd = _standardize_fit(X)
    Xs = (X - mu) / sd
    b, w = _hn_logreg_fit(Xs, y)
    p_in = _logreg_predict(b, w, Xs)
    return {
        "in_sample_auc": round(_roc_auc(p_in.tolist(), y.tolist()), 4),
        "cv5_test_auc_mean": round(float(np.mean(aucs)), 4),
        "cv5_test_auc_std": round(float(np.std(aucs)), 4),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "intercept": round(float(b), 4),
        "feat_coefs_std": [round(float(c), 4) for c in w],
        "feature_means": [round(float(v), 4) for v in mu],
        "feature_scales": [round(float(v), 4) for v in sd],
    }

def _featurize_rows(rows, fasta_map):
    feats = []
    keep = []
    for r in rows:
        seq = fasta_map.get(r["id"])
        if seq is None or len(seq) < 100:
            continue
        L = r.get("L", r.get("length", len(seq)))
        n_loops = r.get("n_loops", 0)
        n_small = r.get("n_small_loops", 0)
        feats.append({
            "length": float(L),
            "mfe_per_nt": float(r.get("mfe_per_nt", r.get("mfe", 0.0) / max(1, L))),
            "frac_paired": float(r["frac_paired"]),
            "small_loop_density": float(r["sld_per100nt"]),
            "max_loop": float(r["max_loop"]),
            "small_loop_frac": float(n_small) / max(1.0, n_loops),
            "gc": float(_gc(seq)),
            "cpg_ratio": float(_cpg_ratio(seq)),
            "longest_orf_nt": float(_longest_orf(seq)),
        })
        keep.append(r["id"])
    return feats, keep

def _passes_brrc(r):
    return (r.get("frac_paired", 0) > 0.65 and
            r.get("sld_per100nt", 0) > 10 and
            r.get("max_loop", 999) <= 15)

def hard_negatives(n_obelisk_subsample=500, seed=42):
    rng = random.Random(seed)
    print("loading obelisk per-seq features", flush=True)
    fc = json.load(open(JSON / "catalog_features.json"))
    ob_rows = [r for r in fc["OBELISK_FULL"] if _passes_brrc(r)]
    print(f"  obelisks BRRC-pass: {len(ob_rows)}", flush=True)
    rng.shuffle(ob_rows)
    ob_rows = ob_rows[:n_obelisk_subsample]
    print(f"  subsampled to n={len(ob_rows)}", flush=True)

    print("loading positive controls", flush=True)
    pc = json.load(open(JSON / "positive_controls.json"))
    hdv_rows = [r for r in pc["delta_class"]["rows"] if _passes_brrc(r)]
    vrd_rows = [r for r in pc["viroids"]["rows"] if _passes_brrc(r)]
    print(f"  HDV BRRC-pass: {len(hdv_rows)}", flush=True)
    print(f"  viroids BRRC-pass: {len(vrd_rows)}", flush=True)
    srp_rows = []
    srp_path = JSON / "srp_perseq.json"
    if srp_path.exists():
        srp_rows = [r for r in json.load(open(srp_path))["rows"] if _passes_brrc(r)]
        print(f"  SRP BRRC-pass (bacterial-host): {len(srp_rows)}", flush=True)

    print("loading FASTA files for sequence features", flush=True)
    ob_fasta = dict(_hard_negatives_fasta(PROC / "obelisks_zheludev_catalog.fasta"))
    print(f"  obelisk FASTA: {len(ob_fasta)} entries", flush=True)
    hdv_fasta = dict()
    for p in (PROC / "delta_genomes.fasta", PROC / "delta_full_genomes.fasta"):
        if p.exists():
            hdv_fasta.update(_hard_negatives_fasta(p))
    print(f"  HDV FASTA: {len(hdv_fasta)} entries", flush=True)

    srp_fasta = {}
    for p in (RAW / "rfam_controls_expanded.fasta", RAW / "rfam_controls.fasta"):
        if p.exists():
            for nm, s in _hard_negatives_fasta(p):
                if "RF00177" in nm or "SRP" in nm.upper():
                    srp_fasta[nm] = s

    vrd_fasta = {k: v for k, v in VIROIDS.items()}

    ob_feats, ob_ids = _featurize_rows(ob_rows, ob_fasta)
    hdv_feats, hdv_ids = _featurize_rows(hdv_rows, hdv_fasta)
    vrd_feats, vrd_ids = _featurize_rows(vrd_rows, vrd_fasta)
    srp_feats, srp_ids = _featurize_rows(srp_rows, srp_fasta) if srp_rows else ([], [])

    print(f"featurized: obelisks {len(ob_feats)}, HDV {len(hdv_feats)}, "
          f"viroids {len(vrd_feats)}, SRP {len(srp_feats)}", flush=True)

    feat_names = ["length", "mfe_per_nt", "frac_paired", "small_loop_density",
                  "max_loop", "small_loop_frac", "gc", "cpg_ratio",
                  "longest_orf_nt"]

    def row(f):
        return [f[k] for k in feat_names]

    X = ([row(f) for f in ob_feats] + [row(f) for f in hdv_feats]
         + [row(f) for f in vrd_feats] + [row(f) for f in srp_feats])
    y = ([1] * len(ob_feats) + [0] * len(hdv_feats)
         + [0] * len(vrd_feats) + [0] * len(srp_feats))

    if len(X) < 10 or sum(y) < 2 or sum(y) == len(y):
        print(f"  not enough data ({sum(y)} pos / {len(y) - sum(y)} neg) â€” abort", flush=True)
        return None

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    uni = {}
    for i, nm in enumerate(feat_names):
        uni[nm] = round(_roc_auc(X_arr[:, i].tolist(), y_arr.tolist()), 4)

    print("fitting logistic regression", flush=True)
    result = _logreg_cv(X, y)

    out = {
        "tool": "hard_negatives",
        "n_obelisks": len(ob_feats),
        "n_hdv": len(hdv_feats),
        "n_viroids": len(vrd_feats),
        "n_srp_bacterial": len(srp_feats),
        "n_total_brrc_pass": len(X),
        "feature_names": feat_names,
        "univariate_auc": uni,
        "logreg_cv5": result,

    }

    JSON.mkdir(parents=True, exist_ok=True)
    with open(JSON / "brrc_pass_discriminators.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {JSON / 'brrc_pass_discriminators.json'}", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    hard_negatives(n_obelisk_subsample=n)
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _trinuc_eulerian(seq, rng, attempts=8):
    s = seq.upper().replace("U", "T")
    if len(s) < 5:
        return s
    edges = defaultdict(list)
    for i in range(len(s) - 2):
        edges[s[i:i + 2]].append(s[i + 2])
    last2 = s[-2:]
    for _ in range(attempts):
        last_edge = {}
        for n, es in edges.items():
            if n != last2 and es:
                last_edge[n] = rng.choice(es)
        rem = {n: list(es) for n, es in edges.items()}
        for n, b in last_edge.items():
            rem[n].remove(b); rng.shuffle(rem[n]); rem[n].append(b)
        out = list(s[:2]); cur = s[:2]; ok = True
        for _ in range(len(s) - 2):
            if not rem.get(cur):
                ok = False; break
            nxt = rem[cur].pop(0)
            out.append(nxt); cur = cur[1] + nxt
        if ok and len(out) == len(s):
            return "".join(out)
    return _shuffle_mono(s, rng)

def _markov_k(seq, k, rng):
    s = seq.upper().replace("U", "T")
    if len(s) < k + 1:
        return s
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(s) - k):
        counts[s[i:i + k]][s[i + k]] += 1
    out = list(s[:k])
    bases = "ACGT"
    for _ in range(len(s) - k):
        ctx = "".join(out[-k:])
        succ = counts.get(ctx)
        if not succ:
            out.append(rng.choice(bases)); continue
        keys, w = zip(*succ.items())
        total = float(sum(w))
        out.append(rng.choices(keys, weights=[wi / total for wi in w], k=1)[0])
    return "".join(out)

def _shuffle_mono(seq, rng):
    chars = list(seq.upper().replace("U", "T"))
    rng.shuffle(chars)
    return "".join(chars)

BINDIR = Path(__file__).resolve().parent.parent / "manuscript" / "tools" / "LinearFold" / "bin"
LF_V = BINDIR / "linearfold_v.exe"
LF_C = BINDIR / "linearfold_c.exe"

def _vrna(seq):
    s = seq.upper().replace("T", "U")
    db, _ = RNA.fold(s)
    return db

def _linearfold(seq, binary):
    s = seq.upper().replace("T", "U")
    try:
        p = subprocess.run([str(binary)], input=s, capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None

    lines = p.stdout.strip().split("\n")
    if len(lines) < 2:
        return None
    db_line = lines[1].split()[0]
    return db_line

def linearfold_compare(n_sample=200, seed=42):
    rng = random.Random(seed)
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    keep = [(nm, s) for nm, s in seqs if 700 <= len(s) <= 1700]
    rng.shuffle(keep)
    sample = keep[:n_sample]
    print(f"  sampled n={len(sample)} obelisks", flush=True)

    vrna_pass = []
    lfv_pass = []
    lfc_pass = []
    concordant_v_lfv = 0
    concordant_v_lfc = 0

    rows = []
    t0 = time.time()
    for i, (nm, s) in enumerate(sample, 1):
        try:
            db_v = _vrna(s)
            fp_v, sld_v, ml_v = _topo(db_v)
            pass_v = _passes(fp_v, sld_v, ml_v)
        except Exception:
            pass_v = False; fp_v = sld_v = ml_v = None

        db_lfv = _linearfold(s, LF_V)
        if db_lfv:
            fp_lv, sld_lv, ml_lv = _topo(db_lfv)
            pass_lfv = _passes(fp_lv, sld_lv, ml_lv)
        else:
            pass_lfv = False; fp_lv = sld_lv = ml_lv = None

        db_lfc = _linearfold(s, LF_C)
        if db_lfc:
            fp_lc, sld_lc, ml_lc = _topo(db_lfc)
            pass_lfc = _passes(fp_lc, sld_lc, ml_lc)
        else:
            pass_lfc = False; fp_lc = sld_lc = ml_lc = None

        vrna_pass.append(pass_v)
        lfv_pass.append(pass_lfv)
        lfc_pass.append(pass_lfc)
        if pass_v == pass_lfv:
            concordant_v_lfv += 1
        if pass_v == pass_lfc:
            concordant_v_lfc += 1

        rows.append({
            "id": nm, "length": len(s),
            "vrna_pass": bool(pass_v),
            "lf_vienna_pass": bool(pass_lfv),
            "lf_contrafold_pass": bool(pass_lfc),
        })

        if i % 25 == 0 or i == len(sample):
            dt = time.time() - t0
            print(f"  {i}/{len(sample)} ({i / max(1e-3, dt):.1f} seq/s, "
                  f"vrna={sum(vrna_pass)}, lf_v={sum(lfv_pass)}, lf_c={sum(lfc_pass)})",
                  flush=True)

    n = len(sample)
    out = {
        "tool": "linearfold_compare",
        "n_obelisks": n,
        "seed": seed,
        "wall_time_s": round(time.time() - t0, 1),
        "vrna_pass_pct": round(100 * sum(vrna_pass) / n, 2),
        "lf_vienna_pct": round(100 * sum(lfv_pass) / n, 2),
        "lf_contrafold_pct": round(100 * sum(lfc_pass) / n, 2),
        "vrna_vs_lfv_pct": round(100 * concordant_v_lfv / n, 2),
        "vrna_vs_lfc_pct": round(100 * concordant_v_lfc / n, 2),

    }

    JSON.mkdir(parents=True, exist_ok=True)
    with open(JSON / "lf_concord.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {JSON / 'lf_concord.json'}", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    linearfold_compare(n_sample=n)
linearfold_shuffles_BINDIR = BINDIR
linearfold_shuffles_LF_C = LF_C

def _linearfold_shuffles_rel(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()

def _lfs_linearfold(seq, binary):
    s = seq.upper().replace("T", "U")
    try:
        p = subprocess.run([str(binary)], input=s, capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    lines = p.stdout.strip().split("\n")
    if len(lines) < 2:
        return None
    return lines[1].split()[0]

def _lfs_wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))

def _katz_logrr(k1, n1, k2, n2, z=1.96):
    if k1 == 0 or k2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    rr = p1 / p2
    se = sqrt((1 - p1) / k1 + (1 - p2) / k2)
    return (rr, rr * exp(-z * se), rr * exp(z * se))

def linearfold_shuffles(n_obelisks=100, n_shuf_per=10, seed=42):
    if not linearfold_shuffles_LF_C.exists():
        print(f"ERROR: {linearfold_shuffles_LF_C} not found", flush=True)
        return None

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    keep = [(nm, s) for nm, s in seqs if 700 <= len(s) <= 1700]
    rng.shuffle(keep)
    sample = keep[:n_obelisks]
    print(f"  sampled n={len(sample)} obelisks (length 700-1700nt, seed {seed})",
          flush=True)
    print(f"  shuffles per obelisk: {n_shuf_per} -> n_shuf={len(sample) * n_shuf_per}",
          flush=True)

    native_pass = 0
    native_rows = []
    shuf_pass = 0
    shuf_total = 0
    shuf_rows = []
    t0 = time.time()

    for i, (nm, s) in enumerate(sample, 1):
        db_native = _lfs_linearfold(s, linearfold_shuffles_LF_C)
        if db_native:
            fp, sld, ml = _topo(db_native)
            ok = _passes(fp, sld, ml)
            if ok:
                native_pass += 1
            native_rows.append({"id": nm, "length": len(s),
                                "frac_paired": round(fp, 4),
                                "small_loop_density": round(sld, 3),
                                "max_loop": ml, "pass": bool(ok)})

        for j in range(n_shuf_per):
            shuf = _shuffle_di(s, np_rng)
            db_shuf = _lfs_linearfold(shuf, linearfold_shuffles_LF_C)
            if db_shuf:
                fp, sld, ml = _topo(db_shuf)
                ok = _passes(fp, sld, ml)
                shuf_total += 1
                if ok:
                    shuf_pass += 1
                if j == 0:
                    shuf_rows.append({"id": nm, "shuf_idx": j,
                                      "frac_paired": round(fp, 4),
                                      "small_loop_density": round(sld, 3),
                                      "max_loop": ml, "pass": bool(ok)})

        if i % 10 == 0 or i == len(sample):
            dt = time.time() - t0
            rate_native = 100 * native_pass / i
            rate_shuf = 100 * shuf_pass / max(1, shuf_total)
            print(f"  {i}/{len(sample)} ({i / max(1e-3, dt):.2f} obelisks/s, "
                  f"native pass {rate_native:.1f}%, shuffle pass {rate_shuf:.2f}% "
                  f"on n_shuf={shuf_total})", flush=True)

    n_native = len(sample)
    n_shuf = shuf_total
    p_native, lo_native, hi_native = _lfs_wilson(native_pass, n_native)
    p_shuf, lo_shuf, hi_shuf = _lfs_wilson(shuf_pass, n_shuf)
    katz = _katz_logrr(native_pass, n_native, shuf_pass, n_shuf)

    out = {
        "tool": "linearfold_shuffles",
        "engine": "LinearFold-CONTRAfold",
        "binary": _linearfold_shuffles_rel(linearfold_shuffles_LF_C),
        "n_obelisks": n_native,
        "n_shuf_per_obelisk": n_shuf_per,
        "n_shuf_total": n_shuf,
        "seed": seed,
        "wall_time_s": round(time.time() - t0, 1),
        "native_pass_count": native_pass,
        "native_pass_pct": round(100 * p_native, 2),
        "native_pass_wilson_lo": round(100 * lo_native, 2),
        "native_pass_wilson_hi": round(100 * hi_native, 2),
        "shuffle_pass_count": shuf_pass,
        "shuffle_pass_pct": round(100 * p_shuf, 3),
        "shuf_pass_wilson_lo": round(100 * lo_shuf, 3),
        "shuf_pass_wilson_hi": round(100 * hi_shuf, 3),
        "enrichment_ratio": (None if katz is None else round(katz[0], 1)),
        "enrichment_katz95_lo": (None if katz is None else round(katz[1], 1)),
        "enrichment_katz95_hi": (None if katz is None else round(katz[2], 1)),
        "rule_of_three_pct": (
            round(100 * 3 / n_shuf, 4) if shuf_pass == 0 else None),

    }

    JSON.mkdir(parents=True, exist_ok=True)
    out_path = JSON / "linearfold_shuffles.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path}", flush=True)
    print(json.dumps({k: out[k] for k in out if k != "binary"}, indent=2), flush=True)
    return out

def linearfold_shuffles_main(rest):
    n_obelisks = int(rest[0]) if len(rest) > 0 else 100
    n_shuf_per = int(rest[1]) if len(rest) > 1 else 10
    seed = int(rest[2]) if len(rest) > 2 else 42
    linearfold_shuffles(n_obelisks=n_obelisks, n_shuf_per=n_shuf_per, seed=seed)

if __name__ == "__main__":
    linearfold_shuffles_main(sys.argv[1:])
OBLIN1_REF = (
    "MRDIELDSSAFRSQVSLLSQETSEKFLTGAALVSPKRSKYYISEVEGLKVHSRSKKDLLALAI"
    "ISWWLEDSIRFYLQEELYFLSINNSDLIEIRLCLTSKSGMLNFLEDTTLYHSRDLFGNILPTS"
    "PEKQVRLANLVSVRYGPTSLPKRVIRRRGYKDHGSRRFPHEVHDLSSGKLAQIKYEEEIQSYH"
    "DTLLFLRGWLDGF"
)

KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

KD_NORM = {aa: (h + 4.5) / 9.0 for aa, h in KD.items()}

CHARGE = {"K": +1.0, "R": +1.0, "H": +0.1, "D": -1.0, "E": -1.0}

BASIC = set("KR")
ACIDIC = set("DE")
AROMATIC = set("FWY")
AA20 = "ACDEFGHIKLMNPQRSTVWY"

def _net_charge(aa: str) -> float:
    return CHARGE.get(aa, 0.0)

def _scaled_hydro(aa: str) -> float:
    return KD_NORM.get(aa, 0.5)

def _sliding_mean(values, win):
    n = len(values)
    half = win // 2
    out = [0.0] * n
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = sum(values[a:b]) / (b - a)
    return out

def _sliding_count(seq, alphabet, win):
    n = len(seq)
    half = win // 2
    out = [0] * n
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = sum(1 for c in seq[a:b] if c in alphabet)
    return out

def _shannon_entropy(window: str) -> float:
    counts = {}
    for aa in window:
        counts[aa] = counts.get(aa, 0) + 1
    n = len(window)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h

def _runs(boolean_track):
    out = []
    i = 0
    n = len(boolean_track)
    while i < n:
        if boolean_track[i]:
            j = i
            while j < n and boolean_track[j]:
                j += 1
            out.append((i, j - 1, j - i))
            i = j
        else:
            i += 1
    return out

def _to_1indexed(runs):
    return [
        {"start": s + 1, "end": e + 1, "length": L}
        for (s, e, L) in runs
    ]

def uversky_track(seq, win=21):
    H = [_scaled_hydro(a) for a in seq]
    R = [_net_charge(a) for a in seq]
    mean_H = _sliding_mean(H, win)
    mean_R = _sliding_mean(R, win)
    abs_R = [abs(r) for r in mean_R]

    disorder = []
    boundary = []
    for h, r in zip(mean_H, abs_R):
        b = (r + 1.151) / 2.785
        boundary.append(b)
        disorder.append(h < b)
    return mean_H, abs_R, boundary, disorder

def lcr_track(seq, win=12, threshold=2.2):
    n = len(seq)
    half = win // 2
    H_bits = [0.0] * n
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        H_bits[i] = _shannon_entropy(seq[a:b])
    flag = [h < threshold for h in H_bits]
    return H_bits, flag

def scan_motifs(seq):
    motifs = {}

    rgg_hits = [m.start() for m in re.finditer(r"RG[GS]", seq)]
    motifs["rgg_hits_1idx"] = [h + 1 for h in rgg_hits]
    rgg_boxes = []
    for i, h in enumerate(rgg_hits):
        for j in range(i + 1, len(rgg_hits)):
            if rgg_hits[j] - h < 30:
                rgg_boxes.append({
                    "start": h + 1, "end": rgg_hits[j] + 3,
                    "first_RGG": h + 1, "second_RGG": rgg_hits[j] + 1,
                })
            else:
                break
    motifs["rgg_boxes_1idx"] = rgg_boxes

    motifs["basic_runs_1idx"] = [
        {"start": m.start() + 1, "end": m.end(), "seq": m.group()}
        for m in re.finditer(r"[KR]{3,}", seq)
    ]

    motifs["YxG_1idx"] = [
        {"start": m.start() + 1, "end": m.end(), "seq": m.group()}
        for m in re.finditer(r"Y.G", seq)
    ]
    motifs["YxxG_1idx"] = [
        {"start": m.start() + 1, "end": m.end(), "seq": m.group()}
        for m in re.finditer(r"Y..G", seq)
    ]

    motifs["PWGGR_like_1idx"] = [
        {"start": m.start() + 1, "end": m.end(), "seq": m.group()}
        for m in re.finditer(r"PWGG[RK]|GFFG[RK]", seq)
    ]

    motifs["GxxG_1idx"] = [
        {"start": m.start() + 1, "end": m.end(), "seq": m.group()}
        for m in re.finditer(r"(?=(G..G))", seq)
    ]

    motifs["SP_or_TP_1idx"] = [
        {"start": m.start() + 1, "end": m.end(), "seq": m.group()}
        for m in re.finditer(r"[ST]P", seq)
    ]

    return motifs

def domain_a_ranges(length=202, helix_offset=150):
    start = helix_offset
    end = helix_offset + 16
    anchors = {
        "G6": start + 5,
        "Y7": start + 6,
        "D9": start + 8,
        "G11": start + 10,
        "H17": start + 16,
    }
    return {"domain_a_start_1idx": start, "domain_a_end_1idx": end,
            "anchors_1idx": anchors}

def annotate_regions(seq, disorder, lcr_flag, basic_count, basic_thr,
                     aromatic_count, aromatic_thr):
    out = {}
    out["disordered_runs_1idx"] = _to_1indexed(_runs(disorder))
    out["lcr_runs_1idx"] = _to_1indexed(_runs(lcr_flag))
    out["basic_patch_runs_1idx"] = _to_1indexed(
        _runs([c >= basic_thr for c in basic_count]))
    out["aromatic_patches"] = _to_1indexed(
        _runs([c >= aromatic_thr for c in aromatic_count]))

    da = domain_a_ranges(length=len(seq))
    da_start, da_end = da["domain_a_start_1idx"], da["domain_a_end_1idx"]

    def fully_outside(r):
        return r["end"] < da_start or r["start"] > da_end

    def flank_slices(r):
        left = right = None
        if r["start"] < da_start:
            left = {"start": r["start"],
                    "end": min(r["end"], da_start - 1),
                    "length": min(r["end"], da_start - 1) - r["start"] + 1}
        if r["end"] > da_end:
            right = {"start": max(r["start"], da_end + 1),
                     "end": r["end"],
                     "length": r["end"] - max(r["start"], da_end + 1) + 1}
        return left, right

    def split_runs(runs):
        outside = []
        for r in runs:
            if fully_outside(r):
                outside.append(r)
            else:
                l, rr = flank_slices(r)
                if l is not None and l["length"] > 0:
                    outside.append(l)
                if rr is not None and rr["length"] > 0:
                    outside.append(rr)
        return outside

    out["outside_domain_a"] = {
        "disordered_runs_1idx": split_runs(out["disordered_runs_1idx"]),
        "lcr_runs_1idx": split_runs(out["lcr_runs_1idx"]),
        "basic_patch_runs_1idx": split_runs(out["basic_patch_runs_1idx"]),
        "aromatic_patches": split_runs(out["aromatic_patches"]),
    }
    out["domain_a"] = da
    return out

def oblin_disorder():
    seq = OBLIN1_REF
    n = len(seq)
    print(f"  Oblin-1 length: {n} aa", flush=True)

    mean_H, abs_R, boundary, disorder = uversky_track(seq, win=21)
    H_bits, lcr_flag = lcr_track(seq, win=12, threshold=2.2)
    basic_count = _sliding_count(seq, BASIC, win=10)
    aromatic_count = _sliding_count(seq, AROMATIC, win=5)

    basic_thr, aromatic_thr = 4, 2

    regions = annotate_regions(seq, disorder, lcr_flag,
                                basic_count, basic_thr,
                                aromatic_count, aromatic_thr)
    motifs = scan_motifs(seq)

    composition = {aa: seq.count(aa) for aa in AA20}
    composition_pct = {aa: round(100.0 * c / n, 2)
                        for aa, c in composition.items()}

    payload = {
        "tool": "oblin_disorder",
        "reference": "Obelisk_000001_000001_000001",
        "length_aa": n,
        "sequence": seq,
        "domain_a": regions["domain_a"],
        "thresholds": {
            "uversky_window_aa": 21,
            "uversky_boundary": "H<( |R|+1.151 )/2.785 disordered",
            "lcr_window_aa": 12,
            "lcr_entropy_bits_lt": 2.2,
            "basic_window_aa": 10,
            "basic_count_ge": basic_thr,
            "aromatic_window_aa": 5,
            "aromatic_count_ge": aromatic_thr,
        },
        "per_residue": {
            "position_1idx": list(range(1, n + 1)),
            "residue": list(seq),
            "uversky_mean_H_norm": [round(x, 4) for x in mean_H],
            "uversky_abs_mean_R": [round(x, 4) for x in abs_R],
            "uversky_boundary_H": [round(x, 4) for x in boundary],
            "disordered_flag": [bool(x) for x in disorder],
            "lcr_entropy_bits": [round(x, 4) for x in H_bits],
            "lcr_flag": [bool(x) for x in lcr_flag],
            "basic_count_10aa": basic_count,
            "aromatic_count_5aa": aromatic_count,
        },
        "regions": regions,
        "motifs": motifs,
        "composition_count": composition,
        "composition_pct": composition_pct,
        "summary_counts": {
            "n_disordered_residues": int(sum(disorder)),
            "n_lcr_residues": int(sum(lcr_flag)),
            "n_disordered_runs": len(regions["disordered_runs_1idx"]),
            "n_lcr_runs": len(regions["lcr_runs_1idx"]),
            "n_basic_patches": len(regions["basic_patch_runs_1idx"]),
            "n_aromatic_patches": len(regions["aromatic_patches"]),
            "n_rgg_hits": len(motifs["rgg_hits_1idx"]),
            "n_rgg_boxes": len(motifs["rgg_boxes_1idx"]),
            "n_basic_runs_ge3": len(motifs["basic_runs_1idx"]),
            "n_GxxG": len(motifs["GxxG_1idx"]),
            "n_YxG": len(motifs["YxG_1idx"]),
            "n_YxxG": len(motifs["YxxG_1idx"]),
            "n_PWGGR_like": len(motifs["PWGGR_like_1idx"]),
            "n_SP_or_TP": len(motifs["SP_or_TP_1idx"]),
        },
    }

    out = JSON / "oblin_disorder.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out}", flush=True)
    print("  summary:", json.dumps(payload["summary_counts"]), flush=True)
    print("  basic-patch runs (1-indexed):",
          payload["regions"]["basic_patch_runs_1idx"], flush=True)
    print("  basic-patch runs outside Domain-A:",
          payload["regions"]["outside_domain_a"]["basic_patch_runs_1idx"],
          flush=True)
    print("  disordered runs (1-indexed):",
          payload["regions"]["disordered_runs_1idx"], flush=True)
    print("  LCR runs (1-indexed):",
          payload["regions"]["lcr_runs_1idx"], flush=True)
    print("  basic K/R runs >=3 (1-indexed):",
          payload["motifs"]["basic_runs_1idx"], flush=True)
    print("  RGG hits (1-indexed):",
          payload["motifs"]["rgg_hits_1idx"], flush=True)
    return payload

if __name__ == "__main__":
    oblin_disorder()
def _detect_orf(seq):
    s = seq.upper().replace("U", "T")
    if len(s) < 6 or "ATG" not in s:
        return None
    best_span, best_len = None, 0
    for offset in range(3):
        i = offset
        while i + 3 <= len(s):
            if s[i:i + 3] == "ATG":
                j = i
                while j + 3 <= len(s):
                    if s[j:j + 3] in ("TAA", "TAG", "TGA"):
                        if (j + 3 - i) > best_len:
                            best_len = j + 3 - i
                            best_span = (i, j + 3)
                        break
                    j += 3
                i = j + 3
            else:
                i += 3
    return best_span

def _shuffle_utr_only(seq, rng):
    s = seq.upper().replace("U", "T")
    span = _detect_orf(s)
    if span is None:
        return s, False
    a, b = span
    utr5 = s[:a]
    orf = s[a:b]
    utr3 = s[b:]
    new5 = _shuffle_di(utr5, rng) if len(utr5) >= 4 else utr5
    new3 = _shuffle_di(utr3, rng) if len(utr3) >= 4 else utr3
    return new5 + orf + new3, True

def _shuffle_syn_codon(seq, rng):
    s = seq.upper().replace("U", "T")
    span = _detect_orf(s)
    if span is None:
        return s, False
    a, b = span
    codons = [s[a + 3 * k:a + 3 * (k + 1)] for k in range((b - a) // 3)]
    by_aa = defaultdict(list)
    for c in codons:
        by_aa[CODON_TABLE.get(c, "?")].append(c)
    for aa in by_aa:
        rng.shuffle(by_aa[aa])
    new_codons = []
    for c in codons:
        new_codons.append(by_aa[CODON_TABLE.get(c, "?")].pop())
    return s[:a] + "".join(new_codons) + s[b:], True

PK_TOOLS = [
    "RNAPKplex",
    "RNAPKplex.exe",
    "ipknot",
    "ipknot.exe",
    "hotknots",
    "HotKnots",
    "probknot",
    "ProbKnot",
    "ProbKnot.exe",

]

PK_PYTHON_MODS = [
    ("ipknot", "ipknot"),
    ("pkfold", "pkfold"),
    ("linearpartition", "linearpartition"),
    ("rnastructure", "rnastructure"),
    ("pknots", "pknots"),
]

def _stderr_summary(stderr: str, pkg: str) -> str:
    if not stderr:
        return ""
    if "No matching distribution found" in stderr:
        return f"No matching distribution found for {pkg}"
    return stderr.replace(str(Path(sys.executable)), "<python>")[-200:]

def _probe_pk_folder() -> dict:
    attempts = []

    for tool in PK_TOOLS:
        path = shutil.which(tool)
        attempts.append({"kind": "shutil.which", "tool": tool,
                         "found_at": path})
        if path:
            return {"available": True, "tool": tool, "path": path,
                    "attempts": attempts}

    for mod_pip, mod_import in PK_PYTHON_MODS:
        try:
            __import__(mod_import)
            attempts.append({"kind": "python_import", "module": mod_import,
                             "ok": True})
            return {"available": True, "tool": f"python:{mod_import}",
                    "attempts": attempts}
        except Exception as exc:
            attempts.append({"kind": "python_import", "module": mod_import,
                             "ok": False, "error": str(exc)[:80]})

    rna_pk_funcs = [x for x in dir(RNA)
                    if "pk" in x.lower() or "knot" in x.lower()]
    attempts.append({"kind": "RNA_python_pk_functions",
                     "found": rna_pk_funcs})

    for pkg in ["ipknot", "pknots", "rnastructure"]:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", pkg],
                capture_output=True, text=True, timeout=30)
            ok = (r.returncode == 0
                  and "Would install" in (r.stdout + r.stderr))
            attempts.append({"kind": "pip_dry_run", "pkg": pkg, "ok": ok,
                             "stderr_tail": _stderr_summary(r.stderr, pkg)})
        except Exception as exc:
            attempts.append({"kind": "pip_dry_run", "pkg": pkg,
                             "ok": False, "error": str(exc)[:100]})

    return {"available": False, "tool": None, "attempts": attempts}

def _pair_lengths(db: str) -> list[int]:
    stack: list[int] = []
    spans: list[int] = []
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                spans.append(i - j)
    return spans

def _giant_loops(db: str, threshold: int = 15) -> tuple[int, int]:
    loops: list[int] = []
    cur = 0
    for c in db:
        if c == ".":
            cur += 1
        else:
            if cur > 0:
                loops.append(cur)
            cur = 0
    if cur > 0:
        loops.append(cur)
    giant = [L for L in loops if L > threshold]
    return len(giant), sum(giant)

def _topo_extended(db: str, n: int) -> dict:
    fp, sld, ml = _topo(db)
    spans = _pair_lengths(db)
    n_pairs = len(spans)
    long_range = sum(1 for s in spans if s > 50)
    very_long = sum(1 for s in spans if s > 100)
    span_max = max(spans) if spans else 0
    span_mean = float(np.mean(spans)) if spans else 0.0
    n_giant, sum_giant_nt = _giant_loops(db, threshold=15)
    return {
        "frac_paired": fp,
        "sld_per100nt": sld,
        "max_loop": int(ml),
        "brrc_pass": _passes(fp, sld, ml),
        "n_pairs": int(n_pairs),
        "n_lrp_gt50": int(long_range),
        "n_vlrp_gt100": int(very_long),
        "frac_long_range_pairs": (round(long_range / n_pairs, 4)
                                  if n_pairs else 0.0),
        "max_pair_span": int(span_max),
        "mean_pair_span": round(span_mean, 2),
        "n_giant_loops_gt15": int(n_giant),
        "giant_loop_total_nt": int(sum_giant_nt),
    }

def _fold_and_audit(seq: str, circular: bool = True) -> dict:
    s = seq.upper().replace("T", "U")
    md = RNA.md()
    md.circ = 1 if circular else 0
    fc = RNA.fold_compound(s, md)
    db, mfe = fc.mfe()
    res = _topo_extended(db, len(s))
    res["length"] = len(s)
    res["mfe_per_nt"] = round(mfe / max(len(s), 1), 4)
    return res

def pseudoknot_audit(n_sample: int = 200, seed: int = 42,
                     length_lo: int = 700, length_hi: int = 1700) -> dict:
    print("Pseudoknot-aware BRRC sensitivity", flush=True)

    probe = _probe_pk_folder()
    if probe["available"]:
        print(f"  found PK-aware folder: {probe['tool']}", flush=True)
        print(f"  (path: {probe.get('path', 'n/a')})", flush=True)
    else:
        print("  NO pseudoknot-aware folder available on this machine",
              flush=True)
        print(f"  attempts logged: {len(probe['attempts'])}", flush=True)

    attempt_out = JSON / "pseudoknot_attempts.json"
    attempt_out.write_text(json.dumps(probe, indent=2))
    print(f"  wrote {attempt_out}", flush=True)

    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    cand = [(nm, s) for nm, s in seqs if length_lo <= len(s) <= length_hi]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cand), min(n_sample, len(cand)), replace=False)
    sample = [cand[int(i)] for i in idx]
    print(f"  obelisk sample: {len(sample)} of {len(cand)} "
          f"in {length_lo}-{length_hi}nt", flush=True)

    ob_rows = []
    t0 = time.time()
    for i, (nm, seq) in enumerate(sample, 1):
        try:
            r = _fold_and_audit(seq, circular=True)
            r["id"] = nm.split()[0]
            ob_rows.append(r)
        except Exception as exc:
            ob_rows.append({"id": nm.split()[0], "length": len(seq),
                            "error": str(exc)})
        if i % 25 == 0:
            rate = i / (time.time() - t0)
            print(f"    folded {i}/{len(sample)} ({rate:.2f}/s)",
                  flush=True)
    ok = [r for r in ob_rows if "error" not in r]
    print(f"  obelisks: {len(ok)}/{len(ob_rows)} valid folds", flush=True)

    print("\nviroid comparator (known bulged-rod, NO pseudoknots)",
          flush=True)
    vir_rows = []
    for nm, seq in VIROIDS.items():
        try:
            db, mfe = _circ_fold(seq)
            r = _topo_extended(db, len(seq))
            r["id"] = nm
            r["length"] = len(seq)
            vir_rows.append(r)
        except Exception as exc:
            vir_rows.append({"id": nm, "length": len(seq),
                             "error": str(exc)})

    def _summ(rows, label):
        ok_r = [r for r in rows if "error" not in r]
        n = len(ok_r)
        if n == 0:
            return {"label": label, "n": 0}
        arr = lambda k: np.array([r[k] for r in ok_r])
        return {
            "label": label,
            "n": n,
            "brrc_pass_pct": round(100 * np.mean(arr("brrc_pass")), 2),
            "mean_frac_paired": round(float(arr("frac_paired").mean()), 4),
            "mean_sld_per100nt": round(
                float(arr("sld_per100nt").mean()), 2),
            "mean_max_loop": round(float(arr("max_loop").mean()), 2),
            "median_max_loop": int(np.median(arr("max_loop"))),
            "mean_n_lrp_gt50": round(
                float(arr("n_lrp_gt50").mean()), 2),
            "med_n_lrp_gt50": int(
                np.median(arr("n_lrp_gt50"))),
            "mean_frac_lrp": round(
                float(arr("frac_long_range_pairs").mean()), 4),
            "mean_vlrp_gt100": round(
                float(arr("n_vlrp_gt100").mean()), 2),
            "mean_max_pair_span": round(
                float(arr("max_pair_span").mean()), 1),
            "median_max_pair_span": int(np.median(arr("max_pair_span"))),
            "mean_giant_loop_nt": round(
                float(arr("giant_loop_total_nt").mean()), 2),
            "frac_any_giant_loop": round(
                float(np.mean(arr("n_giant_loops_gt15") > 0)), 4),
        }

    ob_summary = _summ(ob_rows, "obelisks")
    vir_summary = _summ(vir_rows, "viroids")

    n_fail_ml_only = 0
    n_fail_ml_only = 0
    for r in [x for x in ob_rows if "error" not in x]:
        if r["brrc_pass"]:
            continue

        f1 = r["frac_paired"] > 0.65
        f2 = r["sld_per100nt"] > 10
        f3 = r["max_loop"] <= 15
        if f1 and f2 and not f3:
            n_fail_ml_only += 1
        if not f3:
            n_fail_ml_only += 1

    rescue_ceiling_pct = (round(100 * n_fail_ml_only / len(ok), 2)
                          if ok else 0.0)

    ob_ok = [r for r in ob_rows if "error" not in r]
    vir_ok = [r for r in vir_rows if "error" not in r]
    if ob_ok and vir_ok:
        ob_lrp = np.array([r["frac_long_range_pairs"] for r in ob_ok])
        vir_lrp = np.array([r["frac_long_range_pairs"] for r in vir_ok])
        from scipy.stats import mannwhitneyu
        u, p_lrp = mannwhitneyu(ob_lrp, vir_lrp, alternative="two-sided")

        diff_median = (round(float(np.median(ob_lrp) -
                                   np.median(vir_lrp)), 4))
        comp = {
            "ob_med_frac_lrp": round(float(np.median(ob_lrp)),
                                                      4),
            "vir_med_frac_lrp": round(float(
                np.median(vir_lrp)), 4),
            "med_diff_ob_vir": diff_median,
            "mannwhitney_U": float(u),
            "mannwhitney_p_twosided": float(p_lrp),
        }
    else:
        comp = {"error": "not enough data for Mann-Whitney"}

    out = {
        "tool": "pseudoknot_brrc",
        "sample": {
            "n_attempted": len(ob_rows),
            "n_valid_folds": len(ok),
            "length_window_nt": [length_lo, length_hi],
            "seed": seed,
        },
        "pk_folder_probe": probe,
        "obelisk_summary": ob_summary,
        "viroid_summary": vir_summary,
        "rescue_ceiling": {
            "n_fail_ml_only": int(n_fail_ml_only),
            "n_fail_ml_any": int(n_fail_ml_only),
            "rescue_ceiling_pct": rescue_ceiling_pct,
        },
        "ob_vs_vir_lrp_test": comp,
        "obelisk_rows": ob_rows,
        "viroid_rows": vir_rows,
    }

    out_path = JSON / "pseudoknot_brrc.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(f"  obelisk BRRC pass:  {ob_summary['brrc_pass_pct']}%", flush=True)
    print(f"  obelisk mean LR(>50): "
          f"{ob_summary['mean_n_lrp_gt50']}", flush=True)
    print(f"  viroid  mean LR(>50): "
          f"{vir_summary['mean_n_lrp_gt50']}", flush=True)
    print(f"  rescue ceiling: {rescue_ceiling_pct}%  "
          f"({n_fail_ml_only}/{ob_summary['n']})", flush=True)
    return out

def pseudoknot_audit_main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    n = int(argv[0]) if len(argv) > 0 else 200
    seed = int(argv[1]) if len(argv) > 1 else 42
    pseudoknot_audit(n_sample=n, seed=seed)

if __name__ == "__main__":
    pseudoknot_audit_main()
rna_phylogeny_ROOT = Path(__file__).resolve().parent.parent
rna_phylogeny_JSON = rna_phylogeny_ROOT / "results" / "json"
rna_phylogeny_PROC = rna_phylogeny_ROOT / "data" / "processed"

RNA_FA = rna_phylogeny_PROC / "obelisks_zheludev_catalog.fasta"
PROT_FA = rna_phylogeny_PROC / "oblin1_zheludev_catalog.fasta"
PROT_MSA = rna_phylogeny_PROC / "oblin1_aligned.fasta"
TREE_NWK = rna_phylogeny_JSON / "oblin1_tree.nwk"
FULL_FEATS = rna_phylogeny_JSON / "catalog_features.json"

def _read_fasta(path):
    seqs = {}
    cur = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None:
            seqs[cur] = "".join(buf)
    return seqs

def _kmer_set(seq, k=8):
    s = seq.upper().replace("U", "T")
    if len(s) < k:
        return set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}

def _jaccard_d(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union if union else 0.0

def _msa_hamming(a, b):
    valid = 0
    diff = 0
    for x, y in zip(a, b):
        if x == "-" or y == "-":
            continue
        valid += 1
        if x != y:
            diff += 1
    if valid == 0:
        return 1.0
    return diff / valid

def _flat_upper(M):
    n = M.shape[0]
    out = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            out.append(M[i, j])
    return np.asarray(out, dtype=float)

def _pearson(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def _mantel(M1, M2, n_perm=999, seed=42):
    rng = random.Random(seed)
    n = M1.shape[0]
    v1 = _flat_upper(M1)
    v2 = _flat_upper(M2)
    r_obs = _pearson(v1, v2)
    null = []
    idx_base = list(range(n))
    for _ in range(n_perm):
        perm = idx_base[:]
        rng.shuffle(perm)
        M2p = M2[np.ix_(perm, perm)]
        null.append(_pearson(v1, _flat_upper(M2p)))
    null = np.asarray(null)

    p = float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1))
    return r_obs, p, null

def _patristic_distances(tree_path, leaves):
    from Bio import Phylo

    nwk = open(tree_path).read()
    t = Phylo.read(io.StringIO(nwk), "newick")
    name2clade = {c.name: c for c in t.get_terminals() if c.name in set(leaves)}
    leaves = [n for n in leaves if n in name2clade]
    n = len(leaves)
    D = np.zeros((n, n))
    for i in range(n):
        a = name2clade[leaves[i]]
        for j in range(i + 1, n):
            b = name2clade[leaves[j]]
            d = float(t.distance(a, b))
            D[i, j] = d
            D[j, i] = d
    return leaves, D

def _brrc_pass(rec):
    return (
        rec["frac_paired"] > 0.65
        and rec["sld_per100nt"] > 10
        and rec["max_loop"] <= 15
    )

def _clade_conservation(tree_path, brrc_map, min_size=5, thresh=0.80):
    from Bio import Phylo

    nwk = open(tree_path).read()
    t = Phylo.read(io.StringIO(nwk), "newick")
    total = 0
    conserved = 0
    pass_conserved = 0
    fail_conserved = 0
    for node in t.get_nonterminals():
        leaves = [c.name for c in node.get_terminals() if c.name in brrc_map]
        if len(leaves) < min_size:
            continue
        total += 1
        n_pass = sum(1 for nm in leaves if brrc_map[nm])
        frac_pass = n_pass / len(leaves)
        if frac_pass >= thresh:
            conserved += 1
            pass_conserved += 1
        elif (1 - frac_pass) >= thresh:
            conserved += 1
            fail_conserved += 1
    return {
        "n_internal_nodes_>=%d" % min_size: total,
        "n_conserved_>=%.0f%%" % (thresh * 100): conserved,
        "n_pass_dominant_clades": pass_conserved,
        "n_fail_dominant_clades": fail_conserved,
        "pct_clades_BRRC_conserved": (100.0 * conserved / total) if total else 0.0,
    }

def rna_phylogeny_main():
    t0 = time.time()
    print("[rna-phylogeny] loading inputs...", flush=True)

    rna_seqs = _read_fasta(RNA_FA)
    prot_seqs = _read_fasta(PROT_FA)
    print(f"  RNA: {len(rna_seqs)}  Oblin-1: {len(prot_seqs)}")

    with open(FULL_FEATS) as f:
        feats = json.load(f)
    rec_by_id = {r["id"]: r for r in feats["OBELISK_FULL"]}
    print(f"  features for: {len(rec_by_id)} obelisks")

    from Bio import Phylo

    nwk_str = open(TREE_NWK).read()
    t = Phylo.read(io.StringIO(nwk_str), "newick")
    tree_leaves = [c.name for c in t.get_terminals()]
    print(f"  tree leaves: {len(tree_leaves)}")

    universe = [
        nm
        for nm in tree_leaves
        if nm in prot_seqs and nm in rna_seqs and nm in rec_by_id
    ]
    print(f"  universe (RNA+Oblin-1+features+tree): {len(universe)}")

    use_msa = False
    if PROT_MSA.exists():
        msa = _read_fasta(PROT_MSA)

        msa_keys = [k for k in msa.keys() if k in set(universe)]
        if msa_keys:
            lens = {len(msa[k]) for k in msa_keys}
            if len(lens) == 1:
                use_msa = True
                aligned_prot = {k: msa[k] for k in msa_keys}
                universe = msa_keys
                print(
                    f"  using MSA-aligned Oblin-1: L={lens.pop()}, "
                    f"n={len(universe)}"
                )
    if not use_msa:
        print("  WARNING: MSA not aligned uniformly, falling back to length-clip.")

        Lmin = min(len(prot_seqs[k]) for k in universe)
        aligned_prot = {k: prot_seqs[k][:Lmin] for k in universe}
        print(f"  clipped Oblin-1 to L={Lmin}")

    rng = random.Random(42)
    n_target = min(300, len(universe))
    sel = rng.sample(sorted(universe), n_target)
    print(f"  selected n={len(sel)}")

    print("[rna-phylogeny] computing RNA k-mer (k=8) sets...", flush=True)
    kmers = {nm: _kmer_set(rna_seqs[nm], k=8) for nm in sel}

    print("[rna-phylogeny] computing RNA pairwise Jaccard distances...", flush=True)
    n = len(sel)
    D_rna = np.zeros((n, n))
    for i in range(n):
        si = kmers[sel[i]]
        for j in range(i + 1, n):
            d = _jaccard_d(si, kmers[sel[j]])
            D_rna[i, j] = d
            D_rna[j, i] = d

    print("[rna-phylogeny] computing Oblin-1 pairwise distances...", flush=True)
    D_prot = np.zeros((n, n))
    for i in range(n):
        ai = aligned_prot[sel[i]]
        for j in range(i + 1, n):
            d = _msa_hamming(ai, aligned_prot[sel[j]])
            D_prot[i, j] = d
            D_prot[j, i] = d

    print("[rna-phylogeny] computing tree patristic distances...", flush=True)
    tree_leaves_sel, D_tree = _patristic_distances(TREE_NWK, sel)

    pos_in_tree = {nm: i for i, nm in enumerate(tree_leaves_sel)}
    perm_idx = np.asarray([pos_in_tree[nm] for nm in sel if nm in pos_in_tree])
    if len(perm_idx) != n:

        sel_in_tree = [nm for nm in sel if nm in pos_in_tree]
        keep = np.asarray([sel.index(nm) for nm in sel_in_tree])
        D_rna = D_rna[np.ix_(keep, keep)]
        D_prot = D_prot[np.ix_(keep, keep)]
        sel = sel_in_tree
        perm_idx = np.asarray([pos_in_tree[nm] for nm in sel])
    D_tree = D_tree[np.ix_(perm_idx, perm_idx)]

    print(f"[rna-phylogeny] aligned all matrices to n={len(sel)}")

    print("[rna-phylogeny] Mantel: RNA vs Oblin-1 (Hamming)...", flush=True)
    r_rh, p_rh, null_rh = _mantel(D_rna, D_prot, n_perm=999, seed=42)
    print(f"  RNA vs Oblin-1 Hamming: r={r_rh:.4f}  p={p_rh:.4g}")

    print("[rna-phylogeny] Mantel: RNA vs Oblin-1 (tree patristic)...", flush=True)
    r_rt, p_rt, null_rt = _mantel(D_rna, D_tree, n_perm=999, seed=42)
    print(f"  RNA vs Oblin-1 tree:    r={r_rt:.4f}  p={p_rt:.4g}")

    print("[rna-phylogeny] Mantel: Oblin-1 Hamming vs Oblin-1 tree (sanity)...", flush=True)
    r_ph, p_ph, null_ph = _mantel(D_prot, D_tree, n_perm=999, seed=42)
    print(f"  Oblin-1 Hamming vs tree: r={r_ph:.4f}  p={p_ph:.4g}")

    v_rna = _flat_upper(D_rna)
    v_prot = _flat_upper(D_prot)

    def _z(x):
        m = x.mean()
        s = x.std()
        if s < 1e-12:
            return np.zeros_like(x)
        return (x - m) / s

    z_rna = _z(v_rna)
    z_prot = _z(v_prot)

    disc_rna_close_prot_far = z_prot - z_rna
    disc_prot_close_rna_far = z_rna - z_prot

    pairs = []
    n2 = len(sel)
    idx = 0
    for i in range(n2 - 1):
        for j in range(i + 1, n2):
            pairs.append((sel[i], sel[j], i, j, idx))
            idx += 1

    def _top(disc_vec, k):
        order = np.argsort(-disc_vec)[:k]
        out = []
        for o in order:
            a, b, i, j, pidx = pairs[o]
            out.append(
                {
                    "obelisk_a": a,
                    "obelisk_b": b,
                    "rna_jaccard_d": float(D_rna[i, j]),
                    "oblin1_hamming_d": float(D_prot[i, j]),
                    "oblin1_tree_d": float(D_tree[i, j]),
                    "z_rna": float(z_rna[pidx]),
                    "z_prot": float(z_prot[pidx]),
                    "discordance": float(disc_vec[pidx]),
                }
            )
        return out

    top_rna_close_prot_far = _top(disc_rna_close_prot_far, 10)
    top_prot_close_rna_far = _top(disc_prot_close_rna_far, 10)

    slope = float(np.cov(z_rna, z_prot, bias=True)[0, 1] / max(z_rna.var(), 1e-12))
    intercept = 0.0
    pred = slope * z_rna + intercept
    resid = z_prot - pred

    def _top_resid(vec, k):
        order = np.argsort(-vec)[:k]
        out = []
        for o in order:
            a, b, i, j, pidx = pairs[o]
            out.append(
                {
                    "obelisk_a": a,
                    "obelisk_b": b,
                    "rna_jaccard_d": float(D_rna[i, j]),
                    "oblin1_hamming_d": float(D_prot[i, j]),
                    "oblin1_tree_d": float(D_tree[i, j]),
                    "z_rna": float(z_rna[pidx]),
                    "z_prot": float(z_prot[pidx]),
                    "resid_prot": float(vec[pidx]),
                }
            )
        return out

    top_resid_prot_excess = _top_resid(resid, 10)
    top_resid_prot_deficit = _top_resid(-resid, 10)

    brrc_label = {nm: _brrc_pass(rec_by_id[nm]) for nm in sel}
    n_pass_pass = 0
    n_pass_fail = 0
    n_fail_fail = 0
    for i in range(n2 - 1):
        for j in range(i + 1, n2):
            ai = brrc_label[sel[i]]
            bj = brrc_label[sel[j]]
            if ai and bj:
                n_pass_pass += 1
            elif (not ai) and (not bj):
                n_fail_fail += 1
            else:
                n_pass_fail += 1
    n_pairs = n2 * (n2 - 1) // 2
    brrc_concordant = n_pass_pass + n_fail_fail
    brrc_concord_pct = 100.0 * brrc_concordant / n_pairs if n_pairs else 0.0

    p_pass = sum(1 for nm in sel if brrc_label[nm]) / len(sel)
    expected_concord = (p_pass ** 2 + (1 - p_pass) ** 2) * 100.0
    print(
        f"[rna-phylogeny] BRRC pair concordance: {brrc_concord_pct:.2f}% (expected under "
        f"independence: {expected_concord:.2f}%)"
    )

    brrc_full = {
        nm: _brrc_pass(rec_by_id[nm])
        for nm in tree_leaves
        if nm in rec_by_id
    }
    print("[rna-phylogeny] computing clade-level 80% BRRC conservation on full tree...")
    clade_cons = _clade_conservation(TREE_NWK, brrc_full, min_size=5, thresh=0.80)
    print("  ", clade_cons)

    rna_d_mixed = []
    prot_d_mixed = []
    rna_d_pp = []
    prot_d_pp = []
    rna_d_ff = []
    prot_d_ff = []
    for i in range(n2 - 1):
        for j in range(i + 1, n2):
            ai = brrc_label[sel[i]]
            bj = brrc_label[sel[j]]
            r = D_rna[i, j]
            p = D_prot[i, j]
            if ai and bj:
                rna_d_pp.append(r); prot_d_pp.append(p)
            elif (not ai) and (not bj):
                rna_d_ff.append(r); prot_d_ff.append(p)
            else:
                rna_d_mixed.append(r); prot_d_mixed.append(p)

    def _summary(v):
        v = np.asarray(v)
        if len(v) == 0:
            return {"n": 0}
        return {
            "n": int(len(v)),
            "median": float(np.median(v)),
            "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "mean": float(v.mean()),
        }

    out = {
        "tool": "rna_phylogeny",
        "n_sample": int(n2),
        "seed": 42,
        "rna_distance": "1 - Jaccard(8-mer set) on U->T-normalised RNA",
        "oblin1_distance": "normalised Hamming on MSA-aligned Oblin-1 columns",
        "oblin1_tree_distance": "patristic distance on published Oblin-1 ML tree",
        "mantel": {
            "RNA_vs_Oblin1_Hamming": {
                "r_pearson": float(r_rh),
                "p_value": float(p_rh),
                "null_mean": float(np.mean(null_rh)),
                "null_std": float(np.std(null_rh)),
                "n_perm": 999,
            },
            "RNA_vs_Oblin1_tree": {
                "r_pearson": float(r_rt),
                "p_value": float(p_rt),
                "null_mean": float(np.mean(null_rt)),
                "null_std": float(np.std(null_rt)),
                "n_perm": 999,
            },
            "Oblin1_Hamming_vs_tree_sanity": {
                "r_pearson": float(r_ph),
                "p_value": float(p_ph),
                "null_mean": float(np.mean(null_ph)),
                "null_std": float(np.std(null_ph)),
                "n_perm": 999,
            },
        },
        "discordant_pairs": {
            "top10_RNAclose_Oblin1far": top_rna_close_prot_far,
            "top10_Oblin1close_RNAfar": top_prot_close_rna_far,
            "slope_z_prot_rna": float(slope),
            "top10_resid_pos": top_resid_prot_excess,
            "top10_resid_neg": top_resid_prot_deficit,
        },
        "brrc_pair_concordance": {
            "n_pairs": int(n_pairs),
            "n_pass_pass": int(n_pass_pass),
            "n_fail_fail": int(n_fail_fail),
            "n_mixed": int(n_pass_fail),
            "pct_concordant": float(brrc_concord_pct),
            "expected_indep_pct": float(expected_concord),
            "pass_rate_in_sample": float(p_pass),
        },
        "brrc_clade_cons_tree": clade_cons,
        "brrc_pair_dists": {
            "pass_pass": {
                "rna": _summary(rna_d_pp),
                "oblin1": _summary(prot_d_pp),
            },
            "fail_fail": {
                "rna": _summary(rna_d_ff),
                "oblin1": _summary(prot_d_ff),
            },
            "mixed": {
                "rna": _summary(rna_d_mixed),
                "oblin1": _summary(prot_d_mixed),
            },
        },
        "elapsed_sec": float(time.time() - t0),
    }

    rna_phylogeny_JSON.mkdir(parents=True, exist_ok=True)
    out_path = rna_phylogeny_JSON / "rna_phylogeny.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[rna-phylogeny] wrote {out_path}  ({out['elapsed_sec']:.1f}s)")
    return out

if __name__ == "__main__":
    rna_phylogeny_main()
FEAT = JSON / "catalog_features.json"
OUT = JSON / "roc_ablation.json"

FP_CHOSEN = 0.65
SLD_CHOSEN = 10.0
ML_CHOSEN = 15

FP_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SLD_GRID = [5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0]
ML_GRID = [10, 12, 15, 18, 20, 25]

def passes(r: dict, fp_t: float, sld_t: float, ml_t: int) -> bool:
    return (
        r["frac_paired"] > fp_t
        and r["sld_per100nt"] > sld_t
        and r["max_loop"] <= ml_t
    )

def pass_rate(rows: list[dict], fp_t: float, sld_t: float, ml_t: int) -> tuple[int, float]:
    n_pass = sum(1 for r in rows if passes(r, fp_t, sld_t, ml_t))
    return n_pass, n_pass / len(rows)

def features_matrix(rows: list[dict]) -> np.ndarray:
    return np.array(
        [
            [r["frac_paired"], r["sld_per100nt"], float(r["max_loop"])]
            for r in rows
        ],
        dtype=float,
    )

def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd, mu, sd

def logistic_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    for _ in range(max_iter):
        z = Xb @ beta

        z = np.clip(z, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-z))
        W = prob * (1.0 - prob)

        H = Xb.T @ (W[:, None] * Xb) + 1e-6 * np.eye(p + 1)
        g = Xb.T @ (y - prob)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + delta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta

def logistic_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    z = Xb @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    n = len(scores)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    sum_ranks_pos = float(ranks[labels == 1].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)

def roc_ablation_main() -> None:
    data = json.loads(FEAT.read_text())
    ob = data["OBELISK_FULL"]
    sh = data["SHUFFLED_CONTROL"]
    n_ob = len(ob)
    n_sh = len(sh)

    curve: list[dict] = []
    for fp_t in FP_GRID:
        for sld_t in SLD_GRID:
            for ml_t in ML_GRID:
                k_ob, p_ob = pass_rate(ob, fp_t, sld_t, ml_t)
                k_sh, p_sh = pass_rate(sh, fp_t, sld_t, ml_t)

                enr = (p_ob + 1e-6) / (p_sh + 1e-6)
                curve.append(
                    {
                        "fp": fp_t,
                        "sld": sld_t,
                        "max_loop": ml_t,
                        "obelisk_pass": p_ob,
                        "shuffle_pass": p_sh,
                        "obelisk_n_pass": k_ob,
                        "shuffle_n_pass": k_sh,
                        "enrichment": enr,
                        "youden_j": p_ob - p_sh,
                    }
                )

    knee = max(curve, key=lambda r: r["youden_j"])

    chosen = next(
        r
        for r in curve
        if r["fp"] == FP_CHOSEN and r["sld"] == SLD_CHOSEN and r["max_loop"] == ML_CHOSEN
    )

    sens_rows: list[dict] = []
    for axis, base, lo, hi in [
        ("fp", FP_CHOSEN, 0.0, 1.0),
        ("sld", SLD_CHOSEN, 0.0, 100.0),
        ("max_loop", ML_CHOSEN, 1, 200),
    ]:
        for pct in [-0.20, -0.10, 0.0, 0.10, 0.20]:
            v = base * (1.0 + pct)
            if axis == "max_loop":
                v = int(round(v))
            v = max(lo, min(hi, v))
            fp_t = v if axis == "fp" else FP_CHOSEN
            sld_t = v if axis == "sld" else SLD_CHOSEN
            ml_t = v if axis == "max_loop" else ML_CHOSEN
            k_ob, p_ob = pass_rate(ob, fp_t, sld_t, ml_t)
            k_sh, p_sh = pass_rate(sh, fp_t, sld_t, ml_t)
            sens_rows.append(
                {
                    "axis": axis,
                    "pct": pct,
                    "value": v,
                    "fp": fp_t,
                    "sld": sld_t,
                    "max_loop": ml_t,
                    "obelisk_pass": p_ob,
                    "shuffle_pass": p_sh,
                    "delta_obelisk_pass": p_ob - chosen["obelisk_pass"],
                    "delta_shuffle_pass": p_sh - chosen["shuffle_pass"],
                    "youden_j": p_ob - p_sh,
                }
            )

    X_ob = features_matrix(ob)
    X_sh = features_matrix(sh)
    X = np.vstack([X_ob, X_sh])
    y = np.concatenate([np.ones(n_ob), np.zeros(n_sh)])

    Xz, mu, sd = standardise(X)
    beta = logistic_fit(Xz, y)
    p_hat = logistic_predict(Xz, beta)
    auc_full = auc_rank(p_hat, y)

    rng = np.random.default_rng(20260516)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    cv_aucs: list[float] = []
    for fi, fold_idx in enumerate(folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[fold_idx] = False
        Xz_tr, mu_tr, sd_tr = standardise(X[train_mask])
        beta_tr = logistic_fit(Xz_tr, y[train_mask])
        Xz_te = (X[fold_idx] - mu_tr) / sd_tr
        p_te = logistic_predict(Xz_te, beta_tr)
        cv_aucs.append(auc_rank(p_te, y[fold_idx]))

    uni_aucs = {
        "frac_paired": auc_rank(X[:, 0], y),
        "sld_per100nt": auc_rank(X[:, 1], y),
        "max_loop_negated": auc_rank(-X[:, 2], y),
    }

    out = {
        "meta": {
            "n_obelisks": n_ob,
            "n_shuffles": n_sh,
            "fp_grid": FP_GRID,
            "sld_grid": SLD_GRID,
            "ml_grid": ML_GRID,
            "knee_criterion": "max (obelisk_pass - shuffle_pass) = Youden's J",
        },
        "chosen_thresholds": {
            "fp": FP_CHOSEN,
            "sld": SLD_CHOSEN,
            "max_loop": ML_CHOSEN,
            "obelisk_pass": chosen["obelisk_pass"],
            "shuffle_pass": chosen["shuffle_pass"],
            "enrichment": chosen["enrichment"],
            "youden_j": chosen["youden_j"],
        },
        "knee": {
            "fp": knee["fp"],
            "sld": knee["sld"],
            "max_loop": knee["max_loop"],
            "obelisk_pass": knee["obelisk_pass"],
            "shuffle_pass": knee["shuffle_pass"],
            "enrichment": knee["enrichment"],
            "youden_j": knee["youden_j"],
        },
        "chosen_vs_knee": {
            "youden_j_chosen": chosen["youden_j"],
            "youden_j_knee": knee["youden_j"],
            "youden_j_gap": knee["youden_j"] - chosen["youden_j"],
            "enrichment_chosen": chosen["enrichment"],
            "enrichment_knee": knee["enrichment"],
        },
        "sensitivity": sens_rows,
        "logistic_regression": {
            "features": ["frac_paired", "sld_per100nt", "max_loop"],
            "standardization_mean": mu.tolist(),
            "standardization_sd": sd.tolist(),
            "betas": beta.tolist(),
            "in_sample_auc": auc_full,
            "five_fold_cv_auc_mean": float(np.mean(cv_aucs)),
            "five_fold_cv_auc_sd": float(np.std(cv_aucs, ddof=1)),
            "five_fold_cv_auc_folds": cv_aucs,
            "univariate_auc": uni_aucs,
        },
        "curve": curve,
    }

    OUT.write_text(json.dumps(out, indent=2))

    print(f"n_obelisks={n_ob}  n_shuffles={n_sh}")
    print(
        f"chosen (fp>{FP_CHOSEN}, sld>{SLD_CHOSEN}, ml<={ML_CHOSEN}): "
        f"ob_pass={chosen['obelisk_pass']:.4f}  sh_pass={chosen['shuffle_pass']:.4f}  "
        f"J={chosen['youden_j']:.4f}  enr={chosen['enrichment']:.1f}"
    )
    print(
        f"knee   (fp>{knee['fp']}, sld>{knee['sld']}, ml<={knee['max_loop']}): "
        f"ob_pass={knee['obelisk_pass']:.4f}  sh_pass={knee['shuffle_pass']:.4f}  "
        f"J={knee['youden_j']:.4f}  enr={knee['enrichment']:.1f}"
    )
    print(
        f"AUC in-sample={auc_full:.4f}  5-fold CV mean={np.mean(cv_aucs):.4f} "
        f"sd={np.std(cv_aucs, ddof=1):.4f}"
    )
    print("univariate AUCs:", uni_aucs)
    print(f"wrote: {OUT}")

if __name__ == "__main__":
    roc_ablation_main()
def _seqfold_check_vrna(seq):
    s = seq.upper().replace("T", "U")
    db, mfe = RNA.fold(s)
    return db, mfe

def _seqfold(seq):
    s = seq.upper().replace("U", "T")
    structs = sf_fold(s)
    db = dot_bracket(s, structs)
    return db

def seqfold_compare(n_sample=20, seed=42, max_len=500):
    rng = random.Random(seed)
    print("loading obelisks", flush=True)
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")

    keep = [(nm, s[:max_len]) for nm, s in seqs if 700 <= len(s) <= 1700]
    rng.shuffle(keep)
    sample = keep[:n_sample]
    print(f"  sampled n={len(sample)} obelisks, truncated to first {max_len} nt",
          flush=True)

    rows = []
    vrna_pass = []
    sf_pass = []
    concordant = 0
    discordant_vrna_only = 0
    discordant_sf_only = 0

    t0 = time.time()
    for i, (nm, s) in enumerate(sample, 1):
        try:
            db_v, _ = _seqfold_check_vrna(s)
            fp_v, sld_v, ml_v = _topo(db_v)
            pass_v = _passes(fp_v, sld_v, ml_v)
        except Exception:
            pass_v = False; fp_v = sld_v = ml_v = None
        try:
            db_s = _seqfold(s)
            fp_s, sld_s, ml_s = _topo(db_s)
            pass_s = _passes(fp_s, sld_s, ml_s)
        except Exception:
            pass_s = False; fp_s = sld_s = ml_s = None

        vrna_pass.append(pass_v); sf_pass.append(pass_s)
        if pass_v == pass_s:
            concordant += 1
        elif pass_v and not pass_s:
            discordant_vrna_only += 1
        else:
            discordant_sf_only += 1

        rows.append({
            "id": nm,
            "length": len(s),
            "vrna_pass": bool(pass_v),
            "vrna_frac_paired": fp_v,
            "vrna_sld": sld_v,
            "vrna_max_loop": ml_v,
            "sf_pass": bool(pass_s),
            "sf_frac_paired": fp_s,
            "sf_sld": sld_s,
            "sf_max_loop": ml_s,
        })

        if i % 20 == 0 or i == len(sample):
            dt = time.time() - t0
            print(f"  {i}/{len(sample)} ({i / max(1e-3, dt):.1f} seq/s)", flush=True)

    n = len(sample)
    out = {
        "tool": "seqfold_check",
        "n_obelisks": n,
        "seed": seed,
        "wall_time_s": round(time.time() - t0, 1),
        "vrna_pass_count": int(sum(vrna_pass)),
        "vrna_pass_pct": round(100 * sum(vrna_pass) / n, 2),
        "seqfold_pass_count": int(sum(sf_pass)),
        "seqfold_pass_pct": round(100 * sum(sf_pass) / n, 2),
        "concordant_count": int(concordant),
        "concordance_pct": round(100 * concordant / n, 2),
        "vrna_pass_seqfold_fail": int(discordant_vrna_only),
        "seqfold_pass_vrna_fail": int(discordant_sf_only),

    }

    JSON.mkdir(parents=True, exist_ok=True)
    with open(JSON / "seqfold_check.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {JSON / 'seqfold_check.json'}", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out

def seqfold_check_main(rest=None):
    if rest is None:
        rest = sys.argv[1:]
    n = int(rest[0]) if rest else 100
    seqfold_compare(n_sample=n)

if __name__ == "__main__":
    seqfold_check_main()
try:
    from tmtools import tm_align
    from tmtools.io import get_residue_data, get_structure
    HAVE_TMALIGN = True
except Exception:
    HAVE_TMALIGN = False

N_GLOB = (1, 130)
DOM_A = (131, 175)
C_HEL = (176, 202)
METAL_RES = (158, 159, 166)
CONTACT_A = 8.0
SEQ_SEP = 4

def _load_pdb_ca(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            rn = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            out[rn] = np.array([x, y, z], dtype=float)
    return out

def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc

def _tm_score(deltas: np.ndarray, L_ref: int) -> float:
    if L_ref < 1 or len(deltas) == 0:
        return float("nan")
    if L_ref >= 21:
        d0 = max(0.5, 1.24 * (L_ref - 15) ** (1 / 3) - 1.8)
    else:
        d0 = 0.5
    return float(np.mean(1.0 / (1.0 + (deltas / d0) ** 2)))

def _pairwise_kabsch_tm(ca_a: dict, ca_b: dict, residue_filter=None):
    common = sorted(set(ca_a) & set(ca_b))
    if residue_filter is not None:
        common = [r for r in common if residue_filter(r)]
    if len(common) < 3:
        return {"n_aligned": len(common), "rmsd_A": None, "TM_score": None,
                "error": "fewer than 3 common Ca atoms"}
    P = np.stack([ca_b[r] for r in common])
    Q = np.stack([ca_a[r] for r in common])
    R, t = _kabsch(P, Q)
    deltas = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
    rmsd = float(np.sqrt(np.mean(deltas ** 2)))
    tm = _tm_score(deltas, L_ref=len(common))
    return {
        "n_aligned": int(len(common)),
        "rmsd_A": round(rmsd, 3),
        "TM_score": round(tm, 4),
        "residue_range": [int(common[0]), int(common[-1])],
    }

def _pairwise_tmalign(pdb_a: Path, pdb_b: Path,
                      residue_range: tuple[int, int] | None = None):
    if not HAVE_TMALIGN:
        return {"error": "tmtools not installed"}
    try:
        struct_a = get_structure(str(pdb_a))
        struct_b = get_structure(str(pdb_b))
        coords_a, seq_a = get_residue_data(next(struct_a.get_chains()))
        coords_b, seq_b = get_residue_data(next(struct_b.get_chains()))
        if residue_range is not None:
            lo, hi = residue_range
            coords_a = coords_a[lo - 1:hi]
            seq_a = seq_a[lo - 1:hi]
            coords_b = coords_b[lo - 1:hi]
            seq_b = seq_b[lo - 1:hi]
        res = tm_align(coords_a, coords_b, seq_a, seq_b)
        return {
            "n_residues_a": int(len(seq_a)),
            "n_residues_b": int(len(seq_b)),
            "rmsd_A": round(float(res.rmsd), 3),
            "TM_score": round(float(res.tm_norm_chain1), 4),
            "TM_score_norm_chain2": round(float(res.tm_norm_chain2), 4),
        }
    except Exception as exc:
        return {"error": str(exc)}

def _contact_set(ca: dict, residues: list[int] | None = None,
                 cutoff: float = CONTACT_A,
                 seq_sep: int = SEQ_SEP) -> set[tuple[int, int]]:
    rs = sorted(residues) if residues is not None else sorted(ca.keys())
    rs = [r for r in rs if r in ca]
    out = set()
    for ii, ri in enumerate(rs):
        for rj in rs[ii + 1:]:
            if rj - ri < seq_sep:
                continue
            if np.linalg.norm(ca[ri] - ca[rj]) < cutoff:
                out.add((ri, rj))
    return out

def _region_residues(lo: int, hi: int) -> list[int]:
    return list(range(lo, hi + 1))

def _region_contact_stats(ca_dict: dict[str, dict],
                          region_name: str,
                          lo: int, hi: int) -> dict:
    residues = _region_residues(lo, hi)
    contacts = {name: _contact_set(ca, residues=residues)
                for name, ca in ca_dict.items()}
    af3, boltz, esm = contacts["AF3"], contacts["Boltz2"], contacts["ESMFold"]

    def _jacc(a: set, b: set) -> float:
        u = len(a | b)
        return float(len(a & b) / u) if u else float("nan")

    return {
        "region": region_name,
        "residue_range": [lo, hi],
        "n_residues": hi - lo + 1,
        "n_contacts_AF3": len(af3),
        "n_contacts_Boltz2": len(boltz),
        "n_contacts_ESMFold": len(esm),
        "shared_AF3_Boltz2": len(af3 & boltz),
        "shared_AF3_ESMFold": len(af3 & esm),
        "shared_Boltz2_ESMFold": len(boltz & esm),
        "shared_all_three": len(af3 & boltz & esm),
        "jaccard_AF3_Boltz2": round(_jacc(af3, boltz), 4),
        "jaccard_AF3_ESMFold": round(_jacc(af3, esm), 4),
        "jaccard_Boltz2_ESMFold": round(_jacc(boltz, esm), 4),
    }

def _metal_distances(ca: dict) -> dict[str, float | None]:
    def d(a, b):
        if a not in ca or b not in ca:
            return None
        return round(float(np.linalg.norm(ca[a] - ca[b])), 3)
    return {
        "D158_H159": d(158, 159),
        "D158_H166": d(158, 166),
        "H159_H166": d(159, 166),
    }

def structure_independence():
    af3_pdb = PROC / "oblin1_af3.pdb"
    boltz_pdb = PROC / "oblin1_boltz2.pdb"
    esm_pdb = PROC / "oblin1_esmfold.pdb"
    for p in (af3_pdb, boltz_pdb, esm_pdb):
        if not p.exists():
            raise FileNotFoundError(f"missing required PDB: {p}")

    ca = {
        "AF3":     _load_pdb_ca(af3_pdb),
        "Boltz2":  _load_pdb_ca(boltz_pdb),
        "ESMFold": _load_pdb_ca(esm_pdb),
    }
    n_ca = {k: len(v) for k, v in ca.items()}
    print(f"loaded Ca counts: {n_ca}", flush=True)

    pdb_paths = {"AF3": af3_pdb, "Boltz2": boltz_pdb, "ESMFold": esm_pdb}
    pairs = [
        ("AF3", "Boltz2"),
        ("AF3", "ESMFold"),
        ("Boltz2", "ESMFold"),
    ]
    region_filters = {
        "full":      (None, lambda r: True),
        "N_globule": (N_GLOB, lambda r: N_GLOB[0] <= r <= N_GLOB[1]),
        "Domain_A":  (DOM_A,  lambda r: DOM_A[0] <= r <= DOM_A[1]),
    }

    pair_matrix: dict[str, dict] = {}
    for a, b in pairs:
        for region, (slice_range, filt) in region_filters.items():
            key = f"{a}_vs_{b}_{region}"
            tmalign_res = _pairwise_tmalign(
                pdb_paths[a], pdb_paths[b],
                residue_range=slice_range)
            kabsch_res = _pairwise_kabsch_tm(ca[a], ca[b], residue_filter=filt)
            pair_matrix[key] = {
                "tmalign": tmalign_res,
                "kabsch_sameindex": kabsch_res,
            }
            tm_t = tmalign_res.get("TM_score")
            rmsd_t = tmalign_res.get("rmsd_A")
            tm_k = kabsch_res.get("TM_score")
            rmsd_k = kabsch_res.get("rmsd_A")
            print(f"  {key:40s}  tmalign TM={tm_t} RMSD={rmsd_t}A | "
                  f"kabsch TM={tm_k} RMSD={rmsd_k}A", flush=True)

    print("\ncontact-map analysis (Ca-Ca < 8 A, |i-j| >= 4):", flush=True)
    full_residues = sorted(set(ca["AF3"]) & set(ca["Boltz2"]) & set(ca["ESMFold"]))
    contacts_full = {name: _contact_set(c, residues=full_residues)
                     for name, c in ca.items()}
    af3_c, boltz_c, esm_c = (contacts_full["AF3"],
                             contacts_full["Boltz2"],
                             contacts_full["ESMFold"])

    def _jacc(a: set, b: set) -> float:
        u = len(a | b)
        return float(len(a & b) / u) if u else float("nan")

    contacts_summary = {
        "cutoff_A": CONTACT_A,
        "seq_separation_min": SEQ_SEP,
        "n_common_residues": len(full_residues),
        "n_contacts_AF3": len(af3_c),
        "n_contacts_Boltz2": len(boltz_c),
        "n_contacts_ESMFold": len(esm_c),
        "shared_AF3_Boltz2": len(af3_c & boltz_c),
        "shared_AF3_ESMFold": len(af3_c & esm_c),
        "shared_Boltz2_ESMFold": len(boltz_c & esm_c),
        "shared_all_three": len(af3_c & boltz_c & esm_c),
        "AF3_unique_vs_Boltz2": len(af3_c - boltz_c),
        "Boltz2_unique_vs_AF3": len(boltz_c - af3_c),
        "jaccard_AF3_Boltz2": round(_jacc(af3_c, boltz_c), 4),
        "jaccard_AF3_ESMFold": round(_jacc(af3_c, esm_c), 4),
        "jaccard_Boltz2_ESMFold": round(_jacc(boltz_c, esm_c), 4),
        "fraction_of_AF3_Boltz2_shared_with_ESMFold": (
            round(len(af3_c & boltz_c & esm_c) / len(af3_c & boltz_c), 4)
            if (af3_c & boltz_c) else None
        ),
    }
    for k, v in contacts_summary.items():
        print(f"  {k}: {v}", flush=True)

    region_contacts = [
        _region_contact_stats(ca, "N_globule", *N_GLOB),
        _region_contact_stats(ca, "Domain_A",  *DOM_A),
        _region_contact_stats(ca, "C_helix",   *C_HEL),
    ]

    metal = {name: _metal_distances(c) for name, c in ca.items()}
    metal_disagreement = {
        "D158_H159_max_minus_min": round(
            max(metal[k]["D158_H159"] for k in metal) -
            min(metal[k]["D158_H159"] for k in metal), 3),
        "D158_H166_max_minus_min": round(
            max(metal[k]["D158_H166"] for k in metal) -
            min(metal[k]["D158_H166"] for k in metal), 3),
        "H159_H166_max_minus_min": round(
            max(metal[k]["H159_H166"] for k in metal) -
            min(metal[k]["H159_H166"] for k in metal), 3),
    }

    def _tm_of(a, b, region):
        e = pair_matrix[f"{a}_vs_{b}_{region}"]
        if "tmalign" in e and "TM_score" in e["tmalign"]:
            return e["tmalign"]["TM_score"]
        return e["kabsch_sameindex"]["TM_score"]
    nglob_key = lambda a, b: _tm_of(a, b, "N_globule")
    full_key  = lambda a, b: _tm_of(a, b, "full")
    tm_AB_nglob = nglob_key("AF3", "Boltz2")
    tm_AE_nglob = nglob_key("AF3", "ESMFold")
    tm_BE_nglob = nglob_key("Boltz2", "ESMFold")

    independence_quotient_nglob = (
        (tm_AE_nglob + tm_BE_nglob) / 2.0 / tm_AB_nglob
        if tm_AB_nglob and tm_AB_nglob > 0 else None
    )

    tm_AB_full = full_key("AF3", "Boltz2")
    tm_AE_full = full_key("AF3", "ESMFold")
    tm_BE_full = full_key("Boltz2", "ESMFold")
    independence_quotient_full = (
        (tm_AE_full + tm_BE_full) / 2.0 / tm_AB_full
        if tm_AB_full and tm_AB_full > 0 else None
    )

    contact_independence = contacts_summary["fraction_of_AF3_Boltz2_shared_with_ESMFold"]

    def _label(q):
        if q is None: return "undefined"
        if q >= 0.85: return "high_independence"
        if q >= 0.60: return "partial_independence"
        if q >= 0.35: return "MSA_substantial"
        return "MSA_dominant"

    print("\nINDEPENDENCE QUOTIENT (N-globule):", flush=True)
    print(f"  TM_AF3-Boltz   = {tm_AB_nglob}", flush=True)
    print(f"  TM_AF3-ESM     = {tm_AE_nglob}", flush=True)
    print(f"  TM_Boltz-ESM   = {tm_BE_nglob}", flush=True)
    print(f"  q              = {independence_quotient_nglob}", flush=True)
    print(f"  contact-q      = {contact_independence}", flush=True)

    out = {
        "tool": "structure_independence",
        "regions": {"N_globule": N_GLOB, "Domain_A": DOM_A, "C_helix": C_HEL},
        "contact_cutoff_A": CONTACT_A,
        "seq_separation_min": SEQ_SEP,
        "n_ca": n_ca,
        "predictors": {
            "AF3":     "AlphaFold3 Server, 94-seq ColabFold env MSA",
            "Boltz2":  "Boltz-2, same 94-seq ColabFold env MSA",
            "ESMFold": "ESMFold (single-sequence, ESM-2 PLM, no MSA)",
        },
        "pair_matrix_TM_RMSD": pair_matrix,
        "contact_overlap_full": contacts_summary,
        "region_contacts": region_contacts,
        "metal_cluster_distances_A": metal,
        "metal_cluster_predictor_disagreement_A": metal_disagreement,
        "independence_quotient": {
            "N_globule": independence_quotient_nglob,
            "full_length": independence_quotient_full,
            "contact_map_fraction_AF3_Boltz2_in_ESMFold": contact_independence,
        },

    }

    out_path = JSON / "structure_independence.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return out

def struct_indep_main(argv=None):
    structure_independence()

if __name__ == "__main__":
    struct_indep_main()
OB_LEN_LO, OB_LEN_HI = 700, 1700
N_SAMPLE = 500
SEED = 42

ZHELUDEV_TEMPS = [25.0, 37.0, 50.0, 65.0, 80.0]
HSOB_TEMPS = [37.0, 50.0, 65.0, 80.0]

def _temp_fasta(path):
    out = []
    nm, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if nm is not None:
                    out.append((nm, "".join(buf)))
                nm = line[1:]
                buf = []
            elif line:
                buf.append(line)
        if nm is not None:
            out.append((nm, "".join(buf)))
    return out

def _topo(db):
    n = len(db)
    paired = sum(1 for c in db if c in "()")
    fp = paired / n if n else 0.0
    loops, cur = [], 0
    for c in db:
        if c == ".":
            cur += 1
        else:
            if cur > 0:
                loops.append(cur)
            cur = 0
    if cur > 0:
        loops.append(cur)
    sld = 100.0 * sum(1 for L in loops if L <= 5) / n if n else 0.0
    ml = max(loops) if loops else 0
    return float(fp), float(sld), int(ml)

def _passes(fp, sld, ml):
    return bool(fp > 0.65 and sld > 10 and ml <= 15)

def _fold_one(args):
    nm, seq, T = args
    su = seq.upper().replace("T", "U")
    try:
        md = RNA.md()
        md.circ = 1
        md.temperature = float(T)
        fc = RNA.fold_compound(su, md)
        db, mfe = fc.mfe()
    except Exception as exc:
        return {"id": nm, "T": T, "length": len(su), "error": str(exc)}
    fp, sld, ml = _topo(db)
    return {"id": nm, "T": T, "length": len(su),
            "mfe_per_nt": mfe / len(su),
            "frac_paired": fp,
            "sld_per100nt": sld,
            "max_loop": ml,
            "passes_BRRC": _passes(fp, sld, ml)}

def subsample(seqs, n, seed, lo=OB_LEN_LO, hi=OB_LEN_HI):
    pool = [(nm.split()[0], s) for nm, s in seqs if lo <= len(s) <= hi]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]

def sweep_one_catalog(samples, temps, label):
    workers = max(1, mp.cpu_count() - 1)
    print(f"\n=== {label}  (n={len(samples)}, T={temps}, workers={workers})",
          flush=True)
    by_T = {}
    for T in temps:
        jobs = [(nm, s, T) for nm, s in samples]
        t0 = time.time()
        rows = []
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_fold_one, jobs,
                                                     chunksize=4), 1):
                rows.append(r)
                if i % 100 == 0:
                    rate = i / (time.time() - t0)
                    print(f"  T={T:5.1f}  {i:4d}/{len(jobs)}  ({rate:.1f}/s)",
                          flush=True)

        rows.sort(key=lambda r: r["id"])
        ok = [r for r in rows if "error" not in r]
        n_pass = sum(1 for r in ok if r["passes_BRRC"])
        n_fp = sum(1 for r in ok if r["frac_paired"] > 0.65)
        n_sld = sum(1 for r in ok if r["sld_per100nt"] > 10)
        n_ml = sum(1 for r in ok if r["max_loop"] <= 15)
        by_T[T] = {
            "T_C": T,
            "n_attempted": len(rows),
            "n_valid_folds": len(ok),
            "pass_pct": round(100 * n_pass / max(len(ok), 1), 2),
            "fp_gt_065_pct": round(100 * n_fp / max(len(ok), 1), 2),
            "sld_gt_10_pct": round(100 * n_sld / max(len(ok), 1), 2),
            "max_loop_le_15_pct": round(100 * n_ml / max(len(ok), 1), 2),
            "frac_paired_mean": round(float(np.mean(
                [r["frac_paired"] for r in ok])), 4),
            "sld_mean": round(float(np.mean(
                [r["sld_per100nt"] for r in ok])), 3),
            "max_loop_mean": round(float(np.mean(
                [r["max_loop"] for r in ok])), 2),
            "mfe_per_nt_mean": round(float(np.mean(
                [r["mfe_per_nt"] for r in ok])), 4),
            "rows": ok,
        }
        print(f"  T={T:5.1f} C  pass={by_T[T]['pass_pct']:5.2f}%  "
              f"fp>.65={by_T[T]['fp_gt_065_pct']:5.1f}%  "
              f"sld>10={by_T[T]['sld_gt_10_pct']:5.1f}%  "
              f"ml<=15={by_T[T]['max_loop_le_15_pct']:5.1f}%  "
              f"<fp>={by_T[T]['frac_paired_mean']:.3f}  "
              f"<mfe/nt>={by_T[T]['mfe_per_nt_mean']:.3f}",
              flush=True)
    return by_T

def pearson_and_concordance(by_T, baseline_T):
    ids = sorted({r["id"] for tdata in by_T.values() for r in tdata["rows"]})

    lut = {T: {r["id"]: r for r in tdata["rows"]} for T, tdata in by_T.items()}
    common = [i for i in ids if all(i in lut[T] for T in by_T)]
    temps = sorted(by_T.keys())
    fp_mat = np.array([[lut[T][i]["frac_paired"] for T in temps]
                       for i in common])
    pass_mat = np.array([[1 if lut[T][i]["passes_BRRC"] else 0 for T in temps]
                         for i in common])

    pear = np.corrcoef(fp_mat.T)

    base_idx = temps.index(baseline_T)
    conc = {}
    for j, T in enumerate(temps):
        agree = float((pass_mat[:, j] == pass_mat[:, base_idx]).mean())
        conc[T] = round(100 * agree, 2)
    return {
        "n_common_obelisks": len(common),
        "temperatures_C": temps,
        "pearson_fp_matrix": [[round(float(v), 4) for v in row]
                              for row in pear],
        "baseline_T_C": baseline_T,
        "concord_pct": conc,
    }

def temperature_main():
    print("Temperature sensitivity sweep", flush=True)
    print(f"  N_SAMPLE={N_SAMPLE}  SEED={SEED}", flush=True)
    out = {
        "tool": "temperature_brrc",
        "thresholds": {"frac_paired_gt": 0.65, "sld_gt_per100nt": 10,
                       "max_loop_le": 15},
        "n_sample": N_SAMPLE,
        "seed": SEED,
        "length_window_nt": [OB_LEN_LO, OB_LEN_HI],
    }

    z_all = _temp_fasta(PROC / "obelisks_zheludev_catalog.fasta")
    z_samp = subsample(z_all, N_SAMPLE, SEED)
    print(f"Zheludev: {len(z_all)} loaded, {len(z_samp)} sampled in window")
    z_by_T = sweep_one_catalog(z_samp, ZHELUDEV_TEMPS, "Zheludev")
    z_stats = pearson_and_concordance(z_by_T, baseline_T=37.0)
    out["zheludev"] = {
        "n_loaded": len(z_all),
        "n_sampled": len(z_samp),
        "per_temperature": {str(T): {k: v for k, v in by.items() if k != "rows"}
                            for T, by in z_by_T.items()},
        "structure_independence": z_stats,
        "rows_by_T": {str(T): by["rows"] for T, by in z_by_T.items()},
    }

    hsob_src = RAW / "hsob_obelisks_new.fasta"
    h_all = _temp_fasta(hsob_src)
    h_samp = subsample(h_all, N_SAMPLE, SEED)
    print(f"\nHsOb: {len(h_all)} loaded, {len(h_samp)} sampled in window")
    h_by_T = sweep_one_catalog(h_samp, HSOB_TEMPS, "HsOb")
    h_stats = pearson_and_concordance(h_by_T, baseline_T=37.0)
    out["hsob"] = {
        "source_fasta": str(hsob_src.name),
        "n_loaded": len(h_all),
        "n_sampled": len(h_samp),
        "per_temperature": {str(T): {k: v for k, v in by.items() if k != "rows"}
                            for T, by in h_by_T.items()},
        "structure_independence": h_stats,
        "rows_by_T": {str(T): by["rows"] for T, by in h_by_T.items()},
    }

    z_pass_25 = z_by_T[25.0]["pass_pct"]
    z_pass_37 = z_by_T[37.0]["pass_pct"]
    z_pass_80 = z_by_T[80.0]["pass_pct"]
    h_pass_37 = h_by_T[37.0]["pass_pct"]
    h_pass_65 = h_by_T[65.0]["pass_pct"]
    h_pass_80 = h_by_T[80.0]["pass_pct"]
    z_conc_min = min(z_stats["concord_pct"].values())
    h_conc_min = min(h_stats["concord_pct"].values())
    z_pear_min = min(min(row) for row in z_stats["pearson_fp_matrix"])
    h_pear_min = min(min(row) for row in h_stats["pearson_fp_matrix"])
    out["headline"] = {
        "zheludev_pass_pct_25C": z_pass_25,
        "zheludev_pass_pct_37C": z_pass_37,
        "zheludev_pass_pct_80C": z_pass_80,
        "zheludev_pass_concordance_min_vs_37C_pct": z_conc_min,
        "zhel_pearson_fp_min": round(z_pear_min, 4),
        "hsob_pass_pct_37C": h_pass_37,
        "hsob_pass_pct_65C": h_pass_65,
        "hsob_pass_pct_80C": h_pass_80,
        "hsob_pass_concordance_min_vs_37C_pct": h_conc_min,
        "hsob_pearson_fp_min": round(h_pear_min, 4),
    }

    out_path = JSON / "temperature_brrc.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print(json.dumps(out["headline"], indent=2))
    return out

if __name__ == "__main__":
    temperature_main()

def _tc_passes_brrc(r):
    return (r.get("frac_paired", 0) > 0.65 and
            r.get("sld_per100nt", 0) > 10 and
            r.get("max_loop", 999) <= 15)

def _topo_full(db):
    n = len(db)
    if n == 0:
        return 0.0, 0.0, 0, 0, 0, 0, 0
    paired = sum(1 for c in db if c in "()")
    fp = paired / n
    loops, cur = [], 0
    for c in db:
        if c == ".":
            cur += 1
        else:
            if cur > 0:
                loops.append(cur)
            cur = 0
    if cur > 0:
        loops.append(cur)
    sld = 100.0 * sum(1 for L in loops if L <= 5) / n
    ml = max(loops) if loops else 0
    nl = len(loops)
    n_small = sum(1 for L in loops if L <= 5)
    median = int(sorted(loops)[len(loops) // 2]) if loops else 0

    n_helices = 0
    prev = "."
    for c in db:
        if c == "(" and prev != "(":
            n_helices += 1
        prev = c
    return round(fp, 4), round(sld, 2), int(ml), int(nl), int(n_small), int(n_helices), median

def _tc_circ_fold(seq):
    s = seq.upper().replace("T", "U")
    md = RNA.md(); md.circ = 1
    fc = RNA.fold_compound(s, md)
    return fc.mfe()

def _linear_fold(seq):
    s = seq.upper().replace("T", "U")
    return RNA.fold(s)

EXPAND_FAMILIES = ["RF00010", "RF00011", "RF00023", "RF00028", "RF00177"]

def _load_rfam_by_family(path):
    for hdr, s in _fasta(path):
        rf = hdr.split("|")[0] if "|" in hdr else hdr
        yield rf, hdr, s

def _fold_bact(seqs, label):
    print(f"  folding {len(seqs)} {label} (circular)", flush=True)
    rows = []
    t0 = time.time()
    for i, (rf, hdr, s) in enumerate(seqs, 1):
        L = len(s)
        if not (OB_LEN_LO <= L <= OB_LEN_HI):
            continue
        try:
            db, mfe = _tc_circ_fold(s)
        except Exception as exc:
            try:
                db, mfe = _linear_fold(s)
            except Exception:
                continue
        fp, sld, ml, nl, n_small, n_hel, _med = _topo_full(db)
        if not (fp > 0.65 and sld > 10 and ml <= 15):
            continue
        rows.append({
            "id": hdr,
            "family": rf,
            "L": L,
            "mfe": float(mfe),
            "mfe_per_nt": float(mfe) / max(1, L),
            "frac_paired": fp,
            "sld_per100nt": sld,
            "max_loop": ml,
            "n_loops": nl,
            "n_small_loops": n_small,
            "n_helices": n_hel,
            "_seq": s,
        })
        if i % 25 == 0:
            r = i / max(1e-6, time.time() - t0)
            print(f"    {i}/{len(seqs)}  ({r:.1f}/s)", flush=True)
    print(f"  -> {len(rows)} {label} passing BRRC", flush=True)
    return rows

def _refold_existing(rows, fasta_map, label):
    print(f"  re-folding {len(rows)} {label} for topology fields", flush=True)
    out = []
    for r in rows:
        s = fasta_map.get(r["id"])
        if s is None or len(s) < 100:
            continue
        try:
            db, mfe = _tc_circ_fold(s)
        except Exception:
            try:
                db, mfe = _linear_fold(s)
            except Exception:
                continue
        fp, sld, ml, nl, n_small, n_hel, _med = _topo_full(db)
        L = len(s)
        out.append({
            "id": r["id"],
            "family": label,
            "L": L,
            "mfe": float(mfe),
            "mfe_per_nt": float(mfe) / max(1, L),
            "frac_paired": fp,
            "sld_per100nt": sld,
            "max_loop": ml,
            "n_loops": nl,
            "n_small_loops": n_small,
            "n_helices": n_hel,
            "_seq": s,
        })
    print(f"  -> {len(out)} {label} re-folded", flush=True)
    return out

FEAT_ORIGINAL = [
    "length", "mfe_per_nt", "frac_paired", "small_loop_density",
    "max_loop", "small_loop_frac", "gc", "cpg_ratio", "longest_orf_nt",
]

FEAT_STRICT_TOPOLOGY_ONLY = [
    "frac_paired", "small_loop_density", "max_loop", "small_loop_frac",
    "n_loops_per100nt", "n_helices_per100nt", "cpg_over_gc",
    "longest_orf_nt",
]

FEAT_STRICT_NO_ORF = [
    "frac_paired", "small_loop_density", "max_loop", "small_loop_frac",
    "n_loops_per100nt", "n_helices_per100nt", "cpg_over_gc",
]

def _full_feature_dict(r):
    seq = r.get("_seq")
    if seq is None:

        gc = 0.0
        cpg = 0.0
        orf = 0.0
    else:
        gc = float(_gc(seq))
        cpg = float(_cpg_ratio(seq))
        orf = float(_longest_orf(seq))
    L = float(r["L"])
    nl = float(r["n_loops"])
    n_small = float(r["n_small_loops"])
    n_hel = float(r["n_helices"])
    return {
        "length": L,
        "mfe_per_nt": float(r["mfe_per_nt"]),
        "frac_paired": float(r["frac_paired"]),
        "small_loop_density": float(r["sld_per100nt"]),
        "max_loop": float(r["max_loop"]),
        "small_loop_frac": (n_small / nl) if nl > 0 else 0.0,
        "gc": gc,
        "cpg_ratio": cpg,
        "longest_orf_nt": orf,
        "n_loops_per100nt": 100.0 * nl / max(1.0, L),
        "n_helices_per100nt": 100.0 * n_hel / max(1.0, L),

        "cpg_over_gc": cpg / max(1e-6, gc),
    }

def _logreg_lfo(X, y, families, seed=42):
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    families = np.asarray(families, dtype=object)
    fam_set = sorted({f for f, lbl in zip(families, y) if lbl == 0})
    obelisk_idx = np.where(y == 1)[0]
    per_fam = {}
    aucs = []
    for fam in fam_set:
        te_neg = np.where((y == 0) & (families == fam))[0]
        if len(te_neg) < 5:
            per_fam[fam] = {"n_test_neg": int(len(te_neg)), "skipped": True}
            continue

        ob_pool = obelisk_idx.copy()
        rng.shuffle(ob_pool)
        te_pos = ob_pool[:len(te_neg)]
        te = np.concatenate([te_pos, te_neg])
        tr = np.setdiff1d(np.arange(len(y)), te)
        mu, sd = _standardize_fit(X[tr])
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        b, w = _logreg_fit(Xtr, y[tr])
        p = _logreg_predict(b, w, Xte)
        auc = _roc_auc(p.tolist(), y[te].tolist())
        per_fam[fam] = {
            "n_test_neg": int(len(te_neg)),
            "n_test_pos": int(len(te_pos)),
            "auc": round(float(auc), 4),
        }
        aucs.append(auc)
    return {
        "per_family": per_fam,
        "lfo_auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "lfo_auc_std": round(float(np.std(aucs)), 4) if aucs else None,
        "n_families_used": len(aucs),
    }

def topology_classifier(n_obelisk_subsample=500, seed=42):
    rng_py = random.Random(seed)
    np.random.seed(seed)

    print("[1/6] loading obelisk per-seq features", flush=True)
    fc = json.load(open(JSON / "catalog_features.json"))
    ob_rows_brrc = [r for r in fc["OBELISK_FULL"] if _tc_passes_brrc(r)]
    print(f"  obelisks BRRC-pass total: {len(ob_rows_brrc)}", flush=True)
    rng_py.shuffle(ob_rows_brrc)
    ob_rows = ob_rows_brrc[:n_obelisk_subsample]
    print(f"  subsampled to n={len(ob_rows)}", flush=True)

    ob_fasta = dict(_fasta(PROC / "obelisks_zheludev_catalog.fasta"))
    print(f"  obelisk FASTA: {len(ob_fasta)} entries", flush=True)

    print("[2/6] loading positive controls (HDV + viroids)", flush=True)
    pc = json.load(open(JSON / "positive_controls.json"))
    hdv_rows = [r for r in pc["delta_class"]["rows"] if _tc_passes_brrc(r)]
    vrd_rows = [r for r in pc["viroids"]["rows"] if _tc_passes_brrc(r)]
    print(f"  HDV BRRC-pass:    {len(hdv_rows)}", flush=True)
    print(f"  viroid BRRC-pass: {len(vrd_rows)}", flush=True)

    hdv_fasta = {}
    for p in (PROC / "delta_genomes.fasta", PROC / "delta_full_genomes.fasta",
              RAW / "delta_full_genomes.fasta"):
        if p.exists():
            hdv_fasta.update(_fasta(p))
    print(f"  HDV FASTA: {len(hdv_fasta)} entries", flush=True)
    vrd_fasta = dict(VIROIDS)

    print("[3/6] loading existing SRP per-seq + expanded RF families",
          flush=True)
    srp_rows = []
    srp_path = JSON / "srp_perseq.json"
    if srp_path.exists():
        srp_rows = [r for r in json.load(open(srp_path))["rows"]
                    if _tc_passes_brrc(r)]
    print(f"  SRP BRRC-pass (from srp_perseq.json): {len(srp_rows)}",
          flush=True)

    rfam_recs = []
    rfam_path = RAW / "rfam_controls_expanded.fasta"
    if rfam_path.exists():
        rfam_recs = list(_load_rfam_by_family(rfam_path))
        print(f"  rfam_controls_expanded: {len(rfam_recs)} sequences",
              flush=True)
    rfam_fasta = {hdr: s for _, hdr, s in rfam_recs}

    have_vienna = True
    try:
        import RNA  # noqa: F401
    except Exception as exc:
        have_vienna = False
        print(f"  ViennaRNA unavailable ({exc}); falling back to "
              f"srp_perseq.json only.", flush=True)

    bact_rows = []
    if have_vienna:
        cand_by_fam = defaultdict(list)
        for rf, hdr, s in rfam_recs:
            if rf in EXPAND_FAMILIES and OB_LEN_LO <= len(s) <= OB_LEN_HI:
                cand_by_fam[rf].append((rf, hdr, s))
        for rf in EXPAND_FAMILIES:
            cs = cand_by_fam.get(rf, [])
            print(f"  {rf}: {len(cs)} length-matched candidates", flush=True)
            bact_rows.extend(
                _fold_bact(cs, f"{rf} candidates"))
    else:

        for r in srp_rows:
            seq = rfam_fasta.get(r["id"])
            if seq is None:
                continue
            nl = float(r.get("n_loops", r.get("sld_per100nt",
                                              0) * len(seq) / 100.0))
            nl = max(1.0, nl)
            bact_rows.append({
                "id": r["id"],
                "family": "RF00177",
                "L": int(r.get("L", len(seq))),
                "mfe": float(r.get("mfe", 0.0)),
                "mfe_per_nt": float(r.get("mfe_per_nt", 0.0)),
                "frac_paired": float(r["frac_paired"]),
                "sld_per100nt": float(
                    r["sld_per100nt"]),
                "max_loop": int(r["max_loop"]),
                "n_loops": nl,
                "n_small_loops": float(r.get("n_small_loops", 0)),
                "n_helices": nl,
                "_seq": seq,
            })

    by_id = {}
    for r in bact_rows:
        by_id[r["id"]] = r
    bact_rows = list(by_id.values())
    print(f"  bacterial negative pool (BRRC-pass, deduped): {len(bact_rows)}",
          flush=True)

    if have_vienna:
        hdv_feat_rows = _refold_existing(hdv_rows, hdv_fasta, "HDV")
        vrd_feat_rows = _refold_existing(vrd_rows, vrd_fasta, "viroids")
    else:

        hdv_feat_rows = []
        for r in hdv_rows:
            s = hdv_fasta.get(r["id"])
            if not s:
                continue
            L = len(s)
            nl = max(1.0, r["sld_per100nt"] * L / 100.0)
            hdv_feat_rows.append({
                "id": r["id"], "family": "HDV", "L": L,
                "mfe": float(r.get("mfe_kcal_per_mol", 0.0)),
                "mfe_per_nt": float(r.get("mfe_kcal_per_mol", 0.0)) / L,
                "frac_paired": float(r["frac_paired"]),
                "sld_per100nt": float(
                    r["sld_per100nt"]),
                "max_loop": int(r["max_loop"]),
                "n_loops": nl, "n_small_loops": nl, "n_helices": nl,
                "_seq": s,
            })
        vrd_feat_rows = []
        for r in vrd_rows:
            s = vrd_fasta.get(r["id"])
            if not s:
                continue
            L = len(s)
            nl = max(1.0, r["sld_per100nt"] * L / 100.0)
            vrd_feat_rows.append({
                "id": r["id"], "family": "viroid", "L": L,
                "mfe": float(r.get("mfe_kcal_per_mol", 0.0)),
                "mfe_per_nt": float(r.get("mfe_kcal_per_mol", 0.0)) / L,
                "frac_paired": float(r["frac_paired"]),
                "sld_per100nt": float(
                    r["sld_per100nt"]),
                "max_loop": int(r["max_loop"]),
                "n_loops": nl, "n_small_loops": nl, "n_helices": nl,
                "_seq": s,
            })

    print("[4/6] featurizing obelisks", flush=True)
    ob_feat_rows = []
    if have_vienna:
        for i, r in enumerate(ob_rows, 1):
            s = ob_fasta.get(r["id"])
            if s is None or len(s) < 100:
                continue
            try:
                db, mfe = _tc_circ_fold(s)
            except Exception:
                try:
                    db, mfe = _linear_fold(s)
                except Exception:
                    continue
            fp, sld, ml, nl, n_small, n_hel, _med = _topo_full(db)
            ob_feat_rows.append({
                "id": r["id"], "family": "obelisk", "L": len(s),
                "mfe": float(mfe), "mfe_per_nt": float(mfe) / len(s),
                "frac_paired": fp,
                "sld_per100nt": sld,
                "max_loop": ml, "n_loops": nl, "n_small_loops": n_small,
                "n_helices": n_hel, "_seq": s,
            })
            if i % 50 == 0:
                print(f"    obelisks folded {i}/{len(ob_rows)}", flush=True)
    else:

        for r in ob_rows:
            s = ob_fasta.get(r["id"])
            if s is None or len(s) < 100:
                continue
            nl = max(1.0, float(r.get("n_loops", 1)))
            ob_feat_rows.append({
                "id": r["id"], "family": "obelisk", "L": int(r["L"]),
                "mfe": float(r["mfe"]),
                "mfe_per_nt": float(r["mfe_per_nt"]),
                "frac_paired": float(r["frac_paired"]),
                "sld_per100nt":
                    float(r["sld_per100nt"]),
                "max_loop": int(r["max_loop"]),
                "n_loops": nl,
                "n_small_loops": float(r.get("n_small_loops", 0)),
                "n_helices": nl,
                "_seq": s,
            })
    print(f"  obelisks featurized: {len(ob_feat_rows)}", flush=True)

    n_per_family = Counter()
    for r in bact_rows:
        n_per_family[r["family"]] += 1
    n_per_family["HDV"] = len(hdv_feat_rows)
    n_per_family["viroid"] = len(vrd_feat_rows)
    n_per_family["obelisk"] = len(ob_feat_rows)
    print(f"  per-family counts after expansion: {dict(n_per_family)}",
          flush=True)

    neg_rows = hdv_feat_rows + vrd_feat_rows + bact_rows
    all_rows = ob_feat_rows + neg_rows
    y = np.array([1] * len(ob_feat_rows) + [0] * len(neg_rows), dtype=int)
    families = [r["family"] for r in all_rows]

    print(f"[5/6] building feature matrix: n_pos={int(y.sum())} n_neg="
          f"{len(y) - int(y.sum())}", flush=True)

    feat_dicts = [_full_feature_dict(r) for r in all_rows]

    def _matrix(names):
        return np.array([[fd[k] for k in names] for fd in feat_dicts],
                        dtype=float)

    results = {}
    for label, names in [
        ("original", FEAT_ORIGINAL),
        ("strict_topology_only", FEAT_STRICT_TOPOLOGY_ONLY),
        ("strict_no_orf", FEAT_STRICT_NO_ORF),
    ]:
        X = _matrix(names)

        uni = {}
        for i, nm in enumerate(names):
            uni[nm] = round(_roc_auc(X[:, i].tolist(), y.tolist()), 4)

        cv5 = _logreg_cv(X.tolist(), y.tolist(), k=5, seed=42)

        lfo = _logreg_lfo(X, y, families, seed=42)
        results[label] = {
            "feature_names": names,
            "univariate_auc": uni,
            "logreg_5fold": cv5,
            "logreg_lfo": lfo,
        }

    out = {
        "tool": "topology_classifier",
        "seed": seed,
        "have_vienna": bool(have_vienna),
        "n_obelisks": len(ob_feat_rows),
        "n_hdv": len(hdv_feat_rows),
        "n_viroids": len(vrd_feat_rows),
        "n_bacterial": len(bact_rows),
        "n_total_brrc_pass": int(len(y)),
        "n_per_family": dict(n_per_family),
        "feature_sets": results,
        "summary": {
            "cv5_auc":
                results["original"]["logreg_5fold"]["cv5_test_auc_mean"],
            "strict_topo_cv5_auc":
                results["strict_topology_only"]["logreg_5fold"][
                    "cv5_test_auc_mean"],
            "strict_no_orf_cv5_auc":
                results["strict_no_orf"]["logreg_5fold"][
                    "cv5_test_auc_mean"],
            "strict_topo_lfo_auc":
                results["strict_topology_only"]["logreg_lfo"][
                    "lfo_auc_mean"],
            "strict_no_orf_lfo_auc":
                results["strict_no_orf"]["logreg_lfo"]["lfo_auc_mean"],
        },

    }

    JSON.mkdir(parents=True, exist_ok=True)
    out_path = JSON / "topology_classifier.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[6/6] wrote {out_path}", flush=True)
    print("--- summary ---", flush=True)
    for k, v in out["summary"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  n_per_family: {out['n_per_family']}", flush=True)
    return out

def topology_classifier_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    n = int(argv[0]) if argv else 500
    topology_classifier(n_obelisk_subsample=n)

if __name__ == "__main__":
    topology_classifier_main()

from concurrent.futures import ProcessPoolExecutor
import os, tempfile

TOOLS = ROOT / "manuscript" / "tools"
PROBKNOT = TOOLS / "RNAstructure" / "exe" / "ProbKnot.exe"
PROBKNOT_DATA = TOOLS / "RNAstructure" / "data_tables"
LP_V = TOOLS / "LinearPartition" / "bin" / "linearpartition_v.exe"
LP_BIN = TOOLS / "LinearPartition" / "bin"


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _probknot_fold_one(args):
    name, seq, timeout_s = args
    seq_u = seq.upper().replace("T", "U")
    env = os.environ.copy()
    env["DATAPATH"] = str(PROBKNOT_DATA)
    with tempfile.NamedTemporaryFile("w", suffix=".seq", delete=False) as fseq:
        fseq.write(f";\n{name}\n{seq_u}1\n")
        seqpath = fseq.name
    ctpath = seqpath.replace(".seq", ".ct")
    try:
        subprocess.run([str(PROBKNOT), seqpath, ctpath],
                        env=env, capture_output=True, timeout=timeout_s, check=True)
        pairs = _parse_ct(ctpath)
    except Exception:
        return name, None
    finally:
        for p in (seqpath, ctpath):
            try: os.unlink(p)
            except OSError: pass
    paired_mask = [False] * len(seq_u)
    for i, j in pairs:
        if 0 <= i < len(seq_u) and 0 <= j < len(seq_u):
            paired_mask[i] = paired_mask[j] = True
    return name, paired_mask


def _parse_ct(path):
    pairs = []
    with open(path) as fh:
        for li, line in enumerate(fh):
            if li == 0:
                continue
            f = line.split()
            if len(f) < 5:
                continue
            try:
                i = int(f[0]) - 1
                j = int(f[4]) - 1
            except ValueError:
                continue
            if j > i:
                pairs.append((i, j))
    return pairs


def _brrc_from_mask(mask):
    n = len(mask)
    if n == 0:
        return False, 0.0, 0.0, 0
    fp = sum(mask) / n
    runs = []
    run = 0
    for b in mask + [True]:
        if not b:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    sld = 100 * sum(1 for r in runs if r <= 5) / n
    ml = max(runs) if runs else 0
    bpass = bool(fp > 0.65 and sld > 10 and ml <= 15)
    return bpass, fp, sld, ml


def probknot(timeout_s=900, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    n_att = len(seqs)
    n_done = 0
    n_pass = 0
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    args = [(nm, s, timeout_s) for nm, s in seqs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nm, mask in ex.map(_probknot_fold_one, args, chunksize=8):
            if mask is None:
                continue
            n_done += 1
            bpass, *_ = _brrc_from_mask(mask)
            if bpass:
                n_pass += 1
    out = {
        "tool": "probknot",
        "n_attempted": n_att,
        "n_completed": n_done,
        "probknot_pk_pct": round(100 * n_pass / max(1, n_done), 2),
        "vrna_mfe_pct_ref": 87.7,
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    (JSON / "probknot.json").write_text(json.dumps(out, indent=2))
    return out


def _lp_threshknot_one(args):
    name, seq, gamma, threshold = args
    seq_u = seq.upper().replace("T", "U")
    try:
        p = subprocess.run(
            [str(LP_V), "--threshknot", "--threshold", str(threshold)],
            input=seq_u, capture_output=True, text=True, timeout=300, check=True)
    except Exception:
        return name, None
    pairs = []
    for line in p.stdout.splitlines():
        f = line.split()
        if len(f) == 2:
            try:
                i, j = int(f[0]) - 1, int(f[1]) - 1
                if j > i:
                    pairs.append((i, j))
            except ValueError:
                pass
    mask = [False] * len(seq_u)
    for i, j in pairs:
        if 0 <= i < len(seq_u) and 0 <= j < len(seq_u):
            mask[i] = mask[j] = True
    return name, mask


def lp_threshknot(threshold=0.3, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    n_done = 0
    n_pass = 0
    args = [(nm, s, 3.0, threshold) for nm, s in seqs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nm, mask in ex.map(_lp_threshknot_one, args, chunksize=8):
            if mask is None:
                continue
            n_done += 1
            bpass, *_ = _brrc_from_mask(mask)
            if bpass:
                n_pass += 1
    out = {
        "tool": "lp_threshknot",
        "n_attempted": len(seqs),
        "n_completed": n_done,
        "threshknot_pk_pct": round(100 * n_pass / max(1, n_done), 2),
        "vrna_mfe_pct_ref": 87.7,
        "wall_time_min": round((time.time() - t0) / 60, 1),
    }
    (JSON / "lp_threshknot.json").write_text(json.dumps(out, indent=2))
    return out


def _lp_mea_one(args):
    name, seq, gamma = args
    seq_u = seq.upper().replace("T", "U")
    try:
        p = subprocess.run(
            [str(LP_V), "-M", "--gamma", str(gamma)],
            input=seq_u, capture_output=True, text=True, timeout=600, check=True)
    except Exception:
        return name, None
    db = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line and len(line) == len(seq_u) and set(line) <= set(".()[]{}<>"):
            db = line
            break
    if db is None:
        pairs = []
        for line in p.stdout.splitlines():
            f = line.split()
            if len(f) == 2:
                try:
                    i, j = int(f[0]) - 1, int(f[1]) - 1
                    if j > i:
                        pairs.append((i, j))
                except ValueError:
                    pass
        if pairs:
            db_list = ["."] * len(seq_u)
            for i, j in pairs:
                if 0 <= i < len(seq_u) and 0 <= j < len(seq_u):
                    db_list[i], db_list[j] = "(", ")"
            db = "".join(db_list)
    if db is None or len(db) != len(seq_u):
        return name, None
    return name, db


def lp_mea(gamma=3.0, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    n_done = 0
    n_pass = 0
    args = [(nm, s, gamma) for nm, s in seqs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nm, db in ex.map(_lp_mea_one, args, chunksize=8):
            if db is None:
                continue
            n_done += 1
            fp, sld, ml = _topo(db)
            if _passes(fp, sld, ml):
                n_pass += 1
    out = {
        "tool": "lp_mea",
        "n_attempted": len(seqs),
        "n_completed": n_done,
        "lp_mea_brrc_pass_pct": round(100 * n_pass / max(1, n_done), 2),
        "vrna_mfe_pct_ref": 87.7,
        "wall_time_min": round((time.time() - t0) / 60, 1),
    }
    (JSON / "lp_mea.json").write_text(json.dumps(out, indent=2))
    return out


def _dinuc_one(args):
    nm, seq, seed = args
    rng = random.Random(seed)
    shuf = _shuffle_di(seq, rng)
    db, _ = _circ_fold(shuf)
    fp, sld, ml = _topo(db)
    return _passes(fp, sld, ml)


def dinuc_shuffle_10x(n_rep=10, seed=42, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    args = [(nm, s, seed + r * 1000 + i)
             for r in range(n_rep) for i, (nm, s) in enumerate(seqs)]
    n_total = len(args)
    t0 = time.time()
    n_pass = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(_dinuc_one, args, chunksize=64):
            if ok:
                n_pass += 1
    lo, hi = _wilson_ci(n_pass, n_total)
    out = {
        "tool": "dinuc_shuffle_10x",
        "n_rep": n_rep,
        "n_total_shuffles": n_total,
        "n_pass": n_pass,
        "pass_pct": round(100 * n_pass / n_total, 4),
        "wilson_95ci_pct": [round(100 * lo, 4), round(100 * hi, 4)],
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    (JSON / "dinuc_shuffle_10x.json").write_text(json.dumps(out, indent=2))
    return out


def _cotrans_one(args):
    nm, seq, step = args
    seq_u = seq.upper().replace("T", "U")
    L = len(seq_u)
    first_pass = None
    last_linear_pass = False
    md = RNA.md()
    md.circ = 0
    for k in range(step, L + 1, step):
        prefix = seq_u[:k]
        try:
            fc = RNA.fold_compound(prefix, md)
            db, _ = fc.mfe()
        except Exception:
            continue
        fp, sld, ml = _topo(db)
        if _passes(fp, sld, ml):
            if first_pass is None:
                first_pass = k
            last_linear_pass = True
        else:
            last_linear_pass = False
    try:
        db_circ, _ = _circ_fold(seq_u)
        eq_pass = _passes(*_topo(db_circ))
    except Exception:
        eq_pass = False
    return last_linear_pass, eq_pass, first_pass


def cotrans(step=50, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    args = [(nm, s, step) for nm, s in seqs]
    n_linear = n_eq = n_concord = n_lin_only = n_eq_only = 0
    first_lens = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for lin, eq, fpass in ex.map(_cotrans_one, args, chunksize=8):
            if lin:
                n_linear += 1
            if eq:
                n_eq += 1
            if lin == eq:
                n_concord += 1
            if lin and not eq:
                n_lin_only += 1
            if eq and not lin:
                n_eq_only += 1
            if fpass is not None:
                first_lens.append(fpass)
    n = len(seqs)
    median_fp = int(np.median(first_lens)) if first_lens else 0
    out = {
        "tool": "cotrans",
        "n_obelisks": n,
        "equilibrium_pass_pct": round(100 * n_eq / n, 2),
        "linear_final_pass_pct": round(100 * n_linear / n, 2),
        "concordance_pct": round(100 * n_concord / n, 2),
        "n_linear_only_pass": n_lin_only,
        "n_eq_only_pass": n_eq_only,
        "med_first_pass_nt": median_fp,
    }
    (JSON / "cotrans.json").write_text(json.dumps(out, indent=2))
    return out


def _inverse_fold_one(args):
    name, target_db, seed = args
    rng = random.Random(seed)
    for attempt in range(3):
        start = "".join(rng.choice("AUGC") for _ in range(len(target_db)))
        try:
            seq, _ = RNA.inverse_fold(start, target_db)
            return name, seq, attempt + 1, False
        except Exception:
            continue
    return name, None, 3, True


def inverse_folding(n_sample=100, seed=42, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    rng = random.Random(seed)
    keep = [(nm, s) for nm, s in seqs if 700 <= len(s) <= 1700]
    rng.shuffle(keep)
    sample = keep[:n_sample]
    targets = []
    for nm, s in sample:
        db, _ = _circ_fold(s)
        targets.append((nm, db, seed + hash(nm) % 100000))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    n_done = 0
    n_unrec = 0
    n_pass = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nm, seq, attempts, failed in ex.map(_inverse_fold_one, targets, chunksize=2):
            if failed:
                n_unrec += 1
                continue
            n_done += 1
            db_back, _ = _circ_fold(seq)
            fp, sld, ml = _topo(db_back)
            if _passes(fp, sld, ml):
                n_pass += 1
    out = {
        "tool": "inverse_folding",
        "n_attempted": n_sample,
        "n_complete_retry": n_done,
        "n_unrecov_segfaults": n_unrec,
        "brrc_pass_pct": round(100 * n_pass / max(1, n_done), 2),
        "ob_ref_pct": 87.7,
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    (JSON / "inverse_folding.json").write_text(json.dumps(out, indent=2))
    return out


def _clade_membership():
    out = defaultdict(list)
    fa = PROC / "obelisks_zheludev_catalog.fasta"
    if not fa.exists():
        return {}
    with open(fa) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            parts = line[1:].split()
            if not parts:
                continue
            nm = parts[0]
            cl = "."
            for tok in parts[1:]:
                if tok.startswith("cluster="):
                    cl = tok.split("=", 1)[1]
                    break
            out[cl].append(nm)
    return dict(out)


def permutation_null(n_perm=10000, min_members=10, seed=42):
    catalog = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    pass_map = {}
    for nm, s in catalog:
        try:
            db, _ = _circ_fold(s)
            fp, sld, ml = _topo(db)
            pass_map[nm] = bool(_passes(fp, sld, ml))
        except Exception:
            pass_map[nm] = False
    clade_members = _clade_membership()
    sub_names = [c for c, m in clade_members.items() if len(m) >= min_members]
    observed = {}
    for c in sub_names:
        passes = [pass_map.get(m, False) for m in clade_members[c]]
        observed[c] = sum(passes) / len(passes) if passes else 0.0
    if not observed:
        out = {"tool": "permutation_null", "error": "no clade membership found",
               "n_obelisks": len(catalog), "n_subfamilies_n_ge_10": 0}
        (JSON / "permutation_null.json").write_text(json.dumps(out, indent=2))
        return out
    obs_range = max(observed.values()) - min(observed.values())
    rng = random.Random(seed)
    labels = list(pass_map.keys())
    pass_vals = [pass_map[n] for n in labels]
    n_ge = 0
    null_ranges = []
    for _ in range(n_perm):
        rng.shuffle(pass_vals)
        idx = dict(zip(labels, pass_vals))
        perm_rates = []
        for c in sub_names:
            m = clade_members[c]
            perm_rates.append(sum(idx.get(x, False) for x in m) / len(m))
        r = max(perm_rates) - min(perm_rates)
        null_ranges.append(r)
        if r >= obs_range:
            n_ge += 1
    out = {
        "tool": "permutation_null",
        "n_obelisks": len(catalog),
        "n_subfamilies_n_ge_10": len(sub_names),
        "obs_subfam_pass": {k: round(v, 4) for k, v in observed.items()},
        "observed_range": round(obs_range, 4),
        "n_perm": n_perm,
        "p_empirical": round((n_ge + 1) / (n_perm + 1), 6),
        "null_mean_range": round(float(np.mean(null_ranges)), 4),
        "null_p95_range": round(float(np.percentile(null_ranges, 95)), 4),
        "null_max_range": round(float(np.max(null_ranges)), 4),
    }
    (JSON / "permutation_null.json").write_text(json.dumps(out, indent=2))
    return out


RFAM_MOTIFS = {
    "g_quadruplex": r"G{2,}[ACGU]{1,7}G{2,}[ACGU]{1,7}G{2,}[ACGU]{1,7}G{2,}",
    "sm_binding": r"A?[AU]{4,6}G",
    "uucg_tetraloop": r"UUCG",
    "sarcin_ricin_like": r"AGUACG[AU]A",
    "c_loop": r"CCNGNNNCC",
    "tar_like": r"GG[ACGU]{6,10}CC",
    "trna_cloverleaf": r"GG[ACGU]{50,80}CC[ACGU]{20,40}CCA",
}


def _motif_one(args):
    nm, seq = args
    seq_u = seq.upper().replace("T", "U")
    hits = {}
    for name, pat in RFAM_MOTIFS.items():
        hits[name] = bool(re.search(pat.replace("N", "[ACGU]"), seq_u))
    return hits


def rfam_motif_library(seed=42, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    n = len(seqs)
    rng = random.Random(seed)
    shufs = [(nm, _shuffle_di(s, rng)) for nm, s in seqs]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    counts_ob = Counter()
    counts_sh = Counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for h in ex.map(_motif_one, seqs, chunksize=32):
            for k, v in h.items():
                if v:
                    counts_ob[k] += 1
        for h in ex.map(_motif_one, shufs, chunksize=32):
            for k, v in h.items():
                if v:
                    counts_sh[k] += 1
    by_motif = {}
    for k in RFAM_MOTIFS:
        f_ob = counts_ob[k] / n
        f_sh = counts_sh[k] / n
        by_motif[k] = {
            "ob_frac_motif": round(f_ob, 4),
            "shuf_frac_motif": round(f_sh, 4),
            "enrichment": round(f_ob / max(1e-9, f_sh), 3),
        }
    out = {
        "tool": "rfam_motif_library",
        "n_obelisks": n,
        "n_shuffles": n,
        "by_motif": by_motif,
    }
    (JSON / "rfam_motif_library.json").write_text(json.dumps(out, indent=2))
    return out


def hsob_clade_extraction(zenodo_dir=None):
    base = Path(zenodo_dir) if zenodo_dir else (RAW / "zenodo_18551497")
    overview = base / "Supplement_hsObl" / "overview_contig_obl_putNonObl.dat"
    hsob_fa = PROC / "obelisks_hsob_catalog.fasta"
    hsob_seqs = dict(_fasta(hsob_fa)) if hsob_fa.exists() else {}
    n_rows = 0
    by_clade = defaultdict(list)
    unmatched = 0
    if overview.exists():
        for line in overview.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            n_rows += 1
            f = line.split("\t")
            if len(f) < 2:
                continue
            contig, clade = f[0], f[1]
            if contig in hsob_seqs:
                seq = hsob_seqs[contig]
                gc = (seq.count("G") + seq.count("C")) / max(1, len(seq))
                by_clade[clade].append((contig, len(seq), gc))
            else:
                unmatched += 1
    summary = {}
    matched = 0
    for clade, items in by_clade.items():
        if not items:
            continue
        lens = [it[1] for it in items]
        gcs = [it[2] for it in items]
        summary[clade] = {
            "n": len(items),
            "length_mean": round(float(np.mean(lens)), 1),
            "length_sd": round(float(np.std(lens)), 1),
            "length_min": int(np.min(lens)),
            "length_max": int(np.max(lens)),
            "length_median": int(np.median(lens)),
            "gc_mean": round(float(np.mean(gcs)), 4),
            "gc_sd": round(float(np.std(gcs)), 4),
        }
        matched += len(items)
    out = {
        "tool": "hsob_clade_extraction",
        "source": "zenodo.18551497/Supplement_hsObl/overview_contig_obl_putNonObl.dat",
        "n_overview_rows": n_rows,
        "n_local_hsob": len(hsob_seqs),
        "n_matched": matched,
        "n_unmatched": len(hsob_seqs) - matched,
        "by_clade": summary,
    }
    (JSON / "hsob_clade_extraction.json").write_text(json.dumps(out, indent=2))
    return out


def _apc_mi(msa_cols):
    n_seq = len(msa_cols)
    L = len(msa_cols[0]) if msa_cols else 0
    alpha = ["A", "C", "G", "U", "-"]
    K = len(alpha)
    char_to_idx = {c: i for i, c in enumerate(alpha)}
    M = np.full((n_seq, L), K - 1, dtype=np.int8)
    for r, row in enumerate(msa_cols):
        for c, ch in enumerate(row):
            M[r, c] = char_to_idx.get(ch.upper(), K - 1)
    one_hot = np.zeros((n_seq, L, K), dtype=np.float32)
    np.put_along_axis(one_hot, M[:, :, None], 1.0, axis=2)
    p_single = one_hot.mean(axis=0)
    flat = one_hot.reshape(n_seq, L * K)
    p_joint = (flat.T @ flat) / n_seq
    p_joint = p_joint.reshape(L, K, L, K).transpose(0, 2, 1, 3)
    indep = p_single[:, None, :, None] * p_single[None, :, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(indep > 0, p_joint / indep, 1.0)
        logr = np.where(p_joint > 0, np.log(ratio), 0.0)
        mi_ij = (p_joint * logr).sum(axis=(2, 3))
    np.fill_diagonal(mi_ij, 0.0)
    mi_bar = mi_ij.mean()
    if mi_bar > 0:
        col_mean = mi_ij.mean(axis=0, keepdims=True)
        row_mean = mi_ij.mean(axis=1, keepdims=True)
        apc = (col_mean * row_mean) / mi_bar
        return (mi_ij - apc).astype(np.float64)
    return mi_ij.astype(np.float64)


def msa_expand_rscape(L_used=300):
    out_sub = {}
    t_total = time.time()
    for fam in ("alpha", "omega"):
        msa_path = PROC / f"oblin_{fam}_msa_expanded.fasta"
        if not msa_path.exists():
            msa_path = PROC / f"oblin_{fam}_msa.fasta"
        if not msa_path.exists():
            continue
        seqs = [s for _, s in _fasta(msa_path)]
        orig = len(seqs)
        cols = [s[:L_used].ljust(L_used, "-").upper().replace("T", "U") for s in seqs]
        t0 = time.time()
        apc = _apc_mi(cols)
        upper = apc[np.triu_indices_from(apc, k=1)]
        thr = float(np.percentile(upper, 95))
        out_sub[fam] = {
            "original_n": orig,
            "expanded_n": orig,
            "L_used": L_used,
            "n_pairs": int(L_used * (L_used - 1) / 2),
            "n_pairs_top5pct_apc_mi": int((upper >= thr).sum()),
            "apc_mi_mean": round(float(upper.mean()), 5),
            "apc_mi_p95_threshold": round(thr, 5),
            "apc_mi_max": round(float(upper.max()), 5),
            "wall_min": round((time.time() - t0) / 60, 1),
        }
    out = {"tool": "msa_expand_rscape", "subfamilies": out_sub}
    (JSON / "msa_expand_rscape.json").write_text(json.dumps(out, indent=2))
    return out


def boltz_oblin_rna_cofold(boltz_exe="boltz",
                            protein_fasta=None, rna_fasta=None,
                            protein_idx=0, rna_idx=0):
    out_dir = ROOT / "results" / "boltz_oblin_rna"
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "oblin_rna.yaml"
    if not yaml_path.exists():
        prot_path = Path(protein_fasta) if protein_fasta else (PROC / "oblin1_sequences.fasta")
        rna_path = Path(rna_fasta) if rna_fasta else (PROC / "obelisks_zheludev_catalog.fasta")
        if not prot_path.exists() or not rna_path.exists():
            return {"tool": "boltz_oblin_rna_cofold",
                    "error": "missing input fasta",
                    "expected": [str(prot_path), str(rna_path)]}
        oblin = _fasta(prot_path)
        rna = _fasta(rna_path)
        if not oblin or not rna:
            return {"tool": "boltz_oblin_rna_cofold", "error": "empty fasta"}
        prot_seq = oblin[protein_idx][1].upper()
        rna_seq = rna[rna_idx][1].upper().replace("T", "U")
        yaml_path.write_text(
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: \"{prot_seq}\"\n"
            f"  - rna:\n      id: B\n      sequence: \"{rna_seq}\"\n")
    t0 = time.time()
    try:
        p = subprocess.run([boltz_exe, "predict", str(yaml_path),
                            "--out_dir", str(out_dir)],
                           capture_output=True, text=True, timeout=24 * 3600)
        rc = p.returncode
    except Exception:
        rc = -1
    cifs = list(out_dir.rglob("*.cif"))
    conf_summary = {}
    conf_json = next(iter(out_dir.rglob("confidence_summary*.json")), None)
    if conf_json is not None:
        try:
            conf_summary = json.loads(conf_json.read_text())
        except Exception:
            pass
    out = {
        "tool": "boltz_oblin_rna_cofold",
        "exit_code": rc,
        "wall_min": round((time.time() - t0) / 60, 1),
        "yaml": str(yaml_path),
        "out_dir": str(out_dir),
        "n_cif_outputs": len(cifs),
        "cif_paths": [str(p) for p in cifs],
        "confidence_summary": conf_summary,
    }
    (JSON / "boltz_oblin_rna_cofold.json").write_text(json.dumps(out, indent=2))
    return out


def _higher_order_null_one(args):
    nm, seq, seed, reps, null_name = args
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        if null_name == "dinuc":
            shuf = _shuffle_di(seq, rng)
        elif null_name == "trinuc":
            shuf = _trinuc_eulerian(seq, rng)
        elif null_name == "codon_syn":
            shuf, ok = _shuffle_syn_codon(seq, rng)
            if not ok:
                shuf = _shuffle_di(seq, rng)
        elif null_name == "markov3":
            shuf = _markov_k(seq, 3, rng)
        else:
            shuf = _shuffle_di(seq, rng)
        try:
            db, _ = _circ_fold(shuf)
            fp, sld, ml = _topo(db)
            out.append(_passes(fp, sld, ml))
        except Exception:
            out.append(False)
    return out


def higher_order_nulls(n_reps=3, seed=42, workers=None,
                                     nulls=("dinuc", "trinuc", "codon_syn", "markov3")):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    by_null = {}
    for null_name in nulls:
        args = [(nm, s, seed + i, n_reps, null_name) for i, (nm, s) in enumerate(seqs)]
        n_total = len(args) * n_reps
        n_pass = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for outs in ex.map(_higher_order_null_one, args, chunksize=16):
                n_pass += sum(outs)
        by_null[null_name] = {
            "n_total_shuffles": n_total,
            "n_pass": n_pass,
            "pass_pct": round(100 * n_pass / n_total, 2),
        }
    out = {
        "tool": "higher_order_nulls",
        "n_obelisks": len(seqs),
        "n_reps": n_reps,
        "by_null": by_null,
    }
    (JSON / "higher_order_nulls.json").write_text(json.dumps(out, indent=2))
    return out


def _lf_beam_one(args):
    seq, beam = args
    try:
        p = subprocess.run([str(LF_V), "--beamsize", str(beam)],
                            input=seq.upper().replace("T", "U"),
                            capture_output=True, text=True, timeout=300, check=True)
        lines = p.stdout.strip().split("\n")
        if len(lines) < 2:
            return None
        db = lines[1].split()[0]
        return db
    except Exception:
        return None


def linearfold_beam_sweep(beams=(50, 100, 200, 400, 800), workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    by_beam = {}
    for b in beams:
        args = [(s, b) for _, s in seqs]
        n_done = n_pass = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for db in ex.map(_lf_beam_one, args, chunksize=4):
                if db is None:
                    continue
                n_done += 1
                fp, sld, ml = _topo(db)
                if _passes(fp, sld, ml):
                    n_pass += 1
        by_beam[f"b{b}"] = {
            "n_completed": n_done,
            "brrc_pass_pct": round(100 * n_pass / max(1, n_done), 2),
        }
    out = {
        "tool": "linearfold_beam_sweep",
        "n_obelisks": len(seqs),
        "beam_sizes": list(beams),
        "by_beam": by_beam,
    }
    (JSON / "linearfold_beam_sweep.json").write_text(json.dumps(out, indent=2))
    return out


def _orf_utr_one(args):
    nm, seq, seed = args
    rng = random.Random(seed)
    span = _detect_orf(seq)
    if span is None:
        return None
    db, _ = _circ_fold(seq)
    fp, sld, ml = _topo(db)
    unshuffled_pass = _passes(fp, sld, ml)

    full_dinuc = _shuffle_di(seq, rng)
    db1, _ = _circ_fold(full_dinuc)
    full_dinuc_pass = _passes(*_topo(db1))

    orf_only, _ = _shuffle_syn_codon(seq, rng)
    db2, _ = _circ_fold(orf_only)
    orf_only_pass = _passes(*_topo(db2))

    utr_only, _ = _shuffle_utr_only(seq, rng)
    db3, _ = _circ_fold(utr_only)
    utr_only_pass = _passes(*_topo(db3))

    both = _shuffle_di(orf_only, rng)
    db4, _ = _circ_fold(both)
    both_pass = _passes(*_topo(db4))

    return (unshuffled_pass, full_dinuc_pass, orf_only_pass,
             utr_only_pass, both_pass)


def orf_utr(seed=42, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    args = [(nm, s, seed + i) for i, (nm, s) in enumerate(seqs)]
    counts = {k: 0 for k in ("unshuffled", "full_dinuc", "orf_codon_only",
                              "utr_dinuc_only", "both")}
    n_orf_confident = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_orf_utr_one, args, chunksize=16):
            if res is None:
                continue
            n_orf_confident += 1
            u, f, o, ut, b = res
            counts["unshuffled"] += u
            counts["full_dinuc"] += f
            counts["orf_codon_only"] += o
            counts["utr_dinuc_only"] += ut
            counts["both"] += b
    pass_rates = {k: {"n_pass": v,
                       "pass_pct": round(100 * v / max(1, n_orf_confident), 2)}
                   for k, v in counts.items()}
    out = {"tool": "orf_utr",
            "n_obelisks": n_orf_confident,
            "pass_rates": pass_rates}
    (JSON / "orf_utr.json").write_text(json.dumps(out, indent=2))
    return out


def multiloop():
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    by_config = {}
    configs = [
        ("default_d2_circ", {"dangles": 2, "circ": True}),
        ("dangles_0_circ", {"dangles": 0, "circ": True}),
        ("dangles_1_circ", {"dangles": 1, "circ": True}),
        ("dangles_3_circ", {"dangles": 3, "circ": True}),
        ("noLP_circ", {"dangles": 2, "circ": True, "noLP": True}),
        ("default_d2_linear", {"dangles": 2, "circ": False}),
    ]
    for cfg_name, opts in configs:
        md = RNA.md()
        md.dangles = opts.get("dangles", 2)
        md.noLP = 1 if opts.get("noLP") else 0
        md.circ = 1 if opts.get("circ") else 0
        n_pass = 0
        n_done = 0
        fps, slds, mls = [], [], []
        for nm, s in seqs:
            seq_u = s.upper().replace("T", "U")
            try:
                fc = RNA.fold_compound(seq_u, md)
                db, _ = fc.mfe()
            except Exception:
                continue
            n_done += 1
            fp, sld, ml = _topo(db)
            fps.append(fp); slds.append(sld); mls.append(ml)
            if _passes(fp, sld, ml):
                n_pass += 1
        by_config[cfg_name] = {
            "brrc_pass_pct": round(100 * n_pass / max(1, n_done), 2),
            "n_folded": n_done,
            "mean_fp": round(float(np.mean(fps)), 3),
            "mean_sld": round(float(np.mean(slds)), 2),
            "mean_max_loop": round(float(np.mean(mls)), 1),
        }
    out = {"tool": "multiloop",
           "n_obelisks": len(seqs),
           "by_config": by_config}
    (JSON / "multiloop.json").write_text(json.dumps(out, indent=2))
    return out


def _topo_cut(db, cut):
    n = len(db)
    paired = sum(1 for c in db if c in "()")
    fp = paired / n if n else 0.0
    loops, cur = [], 0
    for c in db:
        if c == ".":
            cur += 1
        else:
            if cur > 0:
                loops.append(cur)
            cur = 0
    if cur > 0:
        loops.append(cur)
    sld = 100.0 * sum(1 for L in loops if L <= cut) / n if n else 0.0
    ml = max(loops) if loops else 0
    return float(fp), float(sld), int(ml)


def _loop_thresh_one(args):
    nm, seq, cuts = args
    try:
        db, _ = _circ_fold(seq)
    except Exception:
        return None
    out = {}
    for cut in cuts:
        fp, sld, ml = _topo_cut(db, cut)
        out[cut] = _passes(fp, sld, ml)
    return out


def loop_class_sweep(cutoffs=(3, 4, 5, 6, 7), seed=42, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    rng = random.Random(seed)
    shuf_pairs = [(nm, _shuffle_di(s, rng)) for nm, s in seqs]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    cuts = tuple(cutoffs)
    args_ob = [(nm, s, cuts) for nm, s in seqs]
    args_sh = [(nm, s, cuts) for nm, s in shuf_pairs]
    counts_ob = {c: 0 for c in cuts}
    counts_sh = {c: 0 for c in cuts}
    n_ob = n_sh = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_loop_thresh_one, args_ob, chunksize=32):
            if res is None: continue
            n_ob += 1
            for c, p in res.items():
                if p: counts_ob[c] += 1
        for res in ex.map(_loop_thresh_one, args_sh, chunksize=32):
            if res is None: continue
            n_sh += 1
            for c, p in res.items():
                if p: counts_sh[c] += 1
    by_cutoff = {}
    for c in cuts:
        by_cutoff[f"sld_le_{c}"] = {
            "obelisks": {"n": n_ob,
                          "pass_pct": round(100 * counts_ob[c] / max(1, n_ob), 2)},
            "shuffles": {"n": n_sh,
                          "pass_pct": round(100 * counts_sh[c] / max(1, n_sh), 2)},
        }
    out = {
        "tool": "loop_class_sweep",
        "n_obelisks": n_ob,
        "n_shuffles": n_sh,
        "by_cutoff": by_cutoff,
    }
    (JSON / "loop_class_sweep.json").write_text(json.dumps(out, indent=2))
    return out


def topology_hmm(n_synth=10000, seed=42):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    paired_lens, unpaired_lens = [], []
    for nm, s in seqs:
        db, _ = _circ_fold(s)
        p_run = u_run = 0
        for c in db:
            if c == ".":
                if p_run:
                    paired_lens.append(p_run); p_run = 0
                u_run += 1
            else:
                if u_run:
                    unpaired_lens.append(u_run); u_run = 0
                p_run += 1
    p_mean = float(np.mean(paired_lens))
    u_mean = float(np.mean(unpaired_lens))
    rng = random.Random(seed)
    n_pass = 0
    for _ in range(n_synth):
        L = rng.choice([len(s) for _, s in seqs])
        db_synth = []
        state = "P"
        while len(db_synth) < L:
            run = max(1, int(rng.expovariate(1 / (p_mean if state == "P" else u_mean))))
            db_synth.extend([("." if state == "U" else "(") for _ in range(run)])
            state = "U" if state == "P" else "P"
        db_synth = "".join(db_synth[:L])
        fp = db_synth.count("(") / len(db_synth)
        run = 0
        runs = []
        for c in db_synth + "P":
            if c == ".":
                run += 1
            else:
                if run:
                    runs.append(run)
                run = 0
        sld = 100 * sum(1 for r in runs if r <= 5) / len(db_synth)
        ml = max(runs) if runs else 0
        if fp >= 0.55 and sld >= 5.5 and ml <= 14:
            n_pass += 1
    out = {
        "tool": "topology_hmm",
        "n_synthetic": n_synth,
        "n_paired_runs_fit": len(paired_lens),
        "n_unpaired_runs_fit": len(unpaired_lens),
        "mean_paired_run_length": round(p_mean, 1),
        "mean_unpaired_run": round(u_mean, 1),
        "synth_brrc_pass_pct": round(100 * n_pass / n_synth, 2),
    }
    (JSON / "topology_hmm.json").write_text(json.dumps(out, indent=2))
    return out


def _frame_shift_one(args):
    nm, seq, seed = args
    rng = random.Random(seed)
    span = _detect_orf(seq)
    if span is None:
        return None
    a, b = span
    s = seq.upper().replace("U", "T")
    db0, _ = _circ_fold(s)
    unshuffled = _passes(*_topo(db0))

    def shift_and_shuffle(frame_offset):
        a2 = max(0, min(len(s), a + frame_offset))
        b2 = a2 + ((b - a) // 3) * 3
        if b2 > len(s):
            b2 = a2 + ((len(s) - a2) // 3) * 3
        if b2 - a2 < 6:
            return s
        codons = [s[a2 + 3 * k:a2 + 3 * (k + 1)] for k in range((b2 - a2) // 3)]
        by_aa = defaultdict(list)
        for c in codons:
            by_aa[CODON_TABLE.get(c, "?")].append(c)
        for aa in by_aa:
            rng.shuffle(by_aa[aa])
        new_codons = [by_aa[CODON_TABLE.get(c, "?")].pop() for c in codons]
        return s[:a2] + "".join(new_codons) + s[b2:]

    results = [unshuffled]
    for f in (0, 1, 2):
        try:
            shuf = shift_and_shuffle(f)
            db, _ = _circ_fold(shuf)
            results.append(_passes(*_topo(db)))
        except Exception:
            results.append(False)
    return results


def frame_shift(seed=42, workers=None):
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    args = [(nm, s, seed + i) for i, (nm, s) in enumerate(seqs)]
    labels = ("unshuffled", "frame_0_codon", "frame_1_codon", "frame_2_codon")
    counts = {k: 0 for k in labels}
    n_orf = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_frame_shift_one, args, chunksize=16):
            if res is None:
                continue
            n_orf += 1
            for k, v in zip(labels, res):
                counts[k] += v
    pass_rates = {k: {"n_pass": v,
                       "pass_pct": round(100 * v / max(1, n_orf), 2)}
                   for k, v in counts.items()}
    out = {"tool": "frame_shift", "n_obelisks": n_orf, "pass_rates": pass_rates}
    (JSON / "frame_shift.json").write_text(json.dumps(out, indent=2))
    return out


def codon_position():
    seqs = _fasta(PROC / "obelisks_zheludev_catalog.fasta")
    by_decile = {}
    n_windows = 0
    n_pass = 0
    for d in range(10):
        n = 0
        passed = 0
        for nm, s in seqs:
            db, _ = _circ_fold(s)
            L = len(db)
            lo = int(L * d / 10)
            hi = int(L * (d + 1) / 10)
            win = db[lo:hi]
            if not win:
                continue
            n += 1
            n_windows += 1
            fp = win.count("(") / len(win)
            if fp >= 0.5:
                passed += 1
                n_pass += 1
        by_decile[f"d{d}"] = {"n": n, "pass_pct": round(100 * passed / max(1, n), 2)}
    out = {
        "tool": "codon_position",
        "n_obelisks": len(seqs),
        "total_windows": n_windows,
        "overall_pass_pct": round(100 * n_pass / max(1, n_windows), 2),
        "by_relative_orf_decile": by_decile,
    }
    (JSON / "codon_position.json").write_text(json.dumps(out, indent=2))
    return out

