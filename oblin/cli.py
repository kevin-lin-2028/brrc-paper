import sys

HELP = """Usage:
  python -m oblin figs
  python -m oblin figs-main [1 4]
  python -m oblin figs-supp [1 4 8]
  python -m oblin stats
  python -m oblin controls positive|bacterial|matched|circular|snapshot
  python -m oblin predict bundle|af3|boltz|foldseek|compare
  python -m oblin rscape dump|parse
  python -m oblin rnap promoter|sixs
  python -m oblin evolve codon|phylo
  python -m oblin high-nulls [reps] [seed]
  python -m oblin hard-negatives [n]
  python -m oblin topology-classifier [n]
  python -m oblin ensemble [n] [samples] [workers]
  python -m oblin motifs [n] [k] [context]
  python -m oblin coupling [n]
  python -m oblin crispr [seed_k] [max_mm]
  python -m oblin rna-mut [n]
  python -m oblin linearfold [n]
  python -m oblin linearfold-shuffles [n_obelisks] [n_shuf_per] [seed]
  python -m oblin seqfold-check
  python -m oblin pseudoknot
  python -m oblin temperature
  python -m oblin archaeal-promoters [n] [seed]
  python -m oblin alternative-descriptors [n]
  python -m oblin roc-ablation
  python -m oblin oblin-disorder
  python -m oblin rna-phylogeny
  python -m oblin structure-independence
  python -m oblin probknot
  python -m oblin lp-threshknot
  python -m oblin lp-mea
  python -m oblin multi10x-dinuc
  python -m oblin cotrans
  python -m oblin inverse-fold-seed
  python -m oblin perm-null-10000
  python -m oblin rfam-motifs
  python -m oblin hsob-clades
  python -m oblin msa-expand
  python -m oblin boltz-cofold
  python -m oblin higher-nulls-full
  python -m oblin lf-beam-sweep
  python -m oblin orf-utr-full
  python -m oblin frame-shift-full
  python -m oblin multiloop-sweep
  python -m oblin loop-class-sweep
  python -m oblin topology-hmm
  python -m oblin codon-position-full
  python -m oblin extval [primary|fallback|auto]
"""

COMMANDS = set()


def _i(r, i, d):
    return int(r[i]) if len(r) > i else d


def _analyses():
    from . import analyses

    return analyses


def _checks():
    from . import checks

    return checks


def _figures():
    from . import figures

    return figures


def _figs(r):
    if r:
        sys.exit("use figs-main or figs-supp for selected panels")
    figs = _figures()
    figs.main([])
    figs.supplement_main([])


DISPATCH = {
    "figs": _figs,
    "figs-main": lambda r: _figures().main(r),
    "figs-supp": lambda r: _figures().supplement_main(r),
    "rscape": lambda r: _analyses().rscape_main(r),
    "predict": lambda r: _analyses().predict_main(r),
    "controls": lambda r: _analyses().controls_main(r),
    "rnap": lambda r: _analyses().rnap_main(r),
    "evolve": lambda r: _analyses().evolve_main(r),
    "stats": lambda r: _analyses().stats_main(r),
    "high-nulls": lambda r: _checks().higher_order_nulls(
        n_reps=_i(r, 0, 3), seed=_i(r, 1, 42)
    ),
    "hard-negatives": lambda r: _checks().hard_negatives(n_obelisk_subsample=_i(r, 0, 500)),
    "topology-classifier": lambda r: _checks().topology_classifier(
        n_obelisk_subsample=_i(r, 0, 500)
    ),
    "ensemble": lambda r: _checks().ensemble_brrc(
        n_sample=_i(r, 0, 200), n_samples_per_seq=_i(r, 1, 30), workers=_i(r, 2, 4)
    ),
    "motifs": lambda r: _analyses().motifs(
        n_obelisk=_i(r, 0, 300), k=_i(r, 1, 6), context=(r[2] if len(r) > 2 else "loop")
    ),
    "coupling": lambda r: _analyses().coupling(n_sample=_i(r, 0, 300)),
    "crispr": lambda r: _analyses().crispr(seed_k=_i(r, 0, 20), max_mismatch=_i(r, 1, 2)),
    "rna-mut": lambda r: _analyses().rna_mut(n_obelisks=_i(r, 0, 10)),
    "linearfold": lambda r: _checks().linearfold_compare(n_sample=_i(r, 0, 200)),
    "linearfold-shuffles": lambda r: _checks().linearfold_shuffles_main(r),
    "seqfold-check": lambda r: _checks().seqfold_check_main(r),
    "structure-independence": lambda r: _checks().structure_independence(),
    "pseudoknot": lambda r: _checks().pseudoknot_audit_main(r),
    "temperature": lambda r: _checks().temperature_main(),
    "archaeal-promoters": lambda r: _checks().archaeal_promoters_main(
        n_sample=_i(r, 0, 500), seed=_i(r, 1, 42)
    ),
    "alternative-descriptors": lambda r: _checks().run(n_target=_i(r, 0, 500)),
    "roc-ablation": lambda r: _checks().roc_ablation_main(),
    "oblin-disorder": lambda r: _checks().oblin_disorder(),
    "rna-phylogeny": lambda r: _checks().rna_phylogeny_main(),
    "probknot": lambda r: _checks().probknot(),
    "lp-threshknot": lambda r: _checks().lp_threshknot(),
    "lp-mea": lambda r: _checks().lp_mea(),
    "multi10x-dinuc": lambda r: _checks().dinuc_shuffle_10x(
        n_rep=_i(r, 0, 10), seed=_i(r, 1, 42)
    ),
    "cotrans": lambda r: _checks().cotrans(step=_i(r, 0, 50)),
    "inverse-fold-seed": lambda r: _checks().inverse_folding(
        n_sample=_i(r, 0, 100), seed=_i(r, 1, 42)
    ),
    "perm-null-10000": lambda r: _checks().permutation_null(
        n_perm=_i(r, 0, 10000), min_members=_i(r, 1, 10), seed=_i(r, 2, 42)
    ),
    "rfam-motifs": lambda r: _checks().rfam_motif_library(seed=_i(r, 0, 42)),
    "hsob-clades": lambda r: _checks().hsob_clade_extraction(),
    "msa-expand": lambda r: _checks().msa_expand_rscape(L_used=_i(r, 0, 300)),
    "boltz-cofold": lambda r: _checks().boltz_oblin_rna_cofold(),
    "higher-nulls-full": lambda r: _checks().higher_order_nulls(
        n_reps=_i(r, 0, 3), seed=_i(r, 1, 42)
    ),
    "lf-beam-sweep": lambda r: _checks().linearfold_beam_sweep(),
    "orf-utr-full": lambda r: _checks().orf_utr(seed=_i(r, 0, 42)),
    "frame-shift-full": lambda r: _checks().frame_shift(seed=_i(r, 0, 42)),
    "multiloop-sweep": lambda r: _checks().multiloop(),
    "loop-class-sweep": lambda r: _checks().loop_class_sweep(
        seed=_i(r, 0, 42)
    ),
    "topology-hmm": lambda r: _checks().topology_hmm(
        n_synth=_i(r, 0, 10000), seed=_i(r, 1, 42)
    ),
    "codon-position-full": lambda r: _checks().codon_position(),
    "extval": lambda r: _analyses().external_validation_main(r),
}

COMMANDS = set(DISPATCH)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    cmd, *rest = argv
    if cmd not in DISPATCH:
        print(HELP, file=sys.stderr)
        sys.exit(2)
    from .core import ensure_runtime_dirs

    ensure_runtime_dirs()
    try:
        DISPATCH[cmd](rest)
    except ModuleNotFoundError as exc:
        print(
            f"error: command {cmd!r} requires an optional dependency that is "
            f"not installed: {exc.name}\n"
            f'install with: pip install -e ".[analysis]"',
            file=sys.stderr,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
