# brrc-paper

Code and processed data for:

**Discovery of a Conserved Bulged-Rod RNA Envelope in Obelisks Supporting a Testable Host-Polymerase Recruitment Hypothesis**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green.svg)](requirements.txt)

## Summary

Obelisks are circular ~1 kb RNAs with no identifiable RNA-dependent RNA polymerase. This repo defines a bulged-rod RNA topology envelope (BRRC), tests it on the Zheludev 2024 catalog and the independent HsOb 2026 catalog, compares it against structured-RNA controls, and analyzes Oblin-1 sequence/structure features behind a falsifiable host-polymerase recruitment hypothesis.

Headline result: 87.7% BRRC pass in 5,169 Zheludev obelisks vs 0.35% in length- and dinucleotide-matched controls, with robustness checks for clustering, thresholds, higher-order nulls, folding mode, folding engine, ribozyme status, and external validation.

## Layout

```
brrc-paper/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- environment.yml
|-- pyproject.toml
|-- oblin/                  # python package (flat modules)
|   |-- __init__.py
|   |-- __main__.py         # `python -m oblin` entry point
|   |-- core.py             # paths + plotting style
|   |-- figures.py          # main + supplement figure builders
|   |-- analyses.py         # primary analyses
|   |-- checks.py           # robustness checks
|   `-- cli.py              # `python -m oblin <cmd>` dispatch
|-- data/
|   |-- raw/                # public inputs
|   `-- processed/          # fasta, msa, a3m, pdb derived from raw
|-- results/
|   |-- json/               # analysis outputs consumed by figures
|   |-- rscape/             # r-scape outputs
|   |-- af3/                # alphafold3 bundle
|   `-- boltz/              # boltz-2 single-chain bundle
`-- manuscript/
    `-- supplement.pdf      # supplementary PDF
```

## Install

```bash
python -m venv venv
source venv/bin/activate          # windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate brrc-analysis
```

## Reproduce figures

The committed `results/json/`, `results/rscape/` and `data/processed/` files are enough to rebuild every figure.

```bash
python -m oblin figs                  # all main + supplement figures
python -m oblin figs-main             # main only
python -m oblin figs-supp             # supplement only
python -m oblin figs-main 1 4         # selected main figures
python -m oblin figs-supp 1 8 10      # selected supplement figures
```

Paths in `oblin/core.py` resolve relative to this folder, so no local edits are needed after cloning.

## Reproduce statistics

```bash
python -m oblin stats
python -m oblin controls positive
python -m oblin controls bacterial
python -m oblin rscape parse
```

Robustness checks live in `oblin/checks.py`. Run `python -m oblin --help` for the full list.

## Data and code availability

All inputs are public. Primary URLs are in the manuscript Data Availability section.

## Author

Kevin Christopher Lin, Great Neck South High School, Great Neck, NY, USA. Correspondence: `kevin.lin.2028@gmail.com`.

## License

MIT. See `LICENSE`.
