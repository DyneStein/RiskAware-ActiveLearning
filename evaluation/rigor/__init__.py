"""
Rigor layer: the evidence artifacts that turn the prototype into a study.

Nothing in here re-trains a model. Every module reads the artifacts the
24-experiment matrix already produced (final checkpoints + per-round CSVs +
pool prediction dumps) and derives the calibration, statistical,
efficiency, ablation, robustness and explainability evidence from them.

See ROADMAP.md §5 and analysis/rigor/ for the outputs.
"""
