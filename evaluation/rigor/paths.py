"""
Shared paths + constants for the rigor layer.

The rigor layer reads the artifacts the 24 experiments already produced
(checkpoints + per-round CSVs) and writes new evidence artifacts. It never
re-trains anything, so it can run on a laptop CPU.

PATH LAYOUT
-----------
Two roots, because they have different lifetimes and different owners:

    PROJECT_ROOT  the repository — code, results, analysis, paper. Every
                  artefact we produce lives under here and is version
                  controlled (checkpoints excepted; see .gitignore).
                  Defaults to the repo this file sits in, so a fresh
                  clone works with no configuration.

    DATA_ROOT     where the HAM10000 images are. Deliberately OUTSIDE the
                  repository: 2.8 GB, licensed CC BY-NC-SA 4.0, and not
                  ours to redistribute. Defaults to a sibling `archive/`
                  folder next to the repo.

Both can be overridden by environment variables of the same name, which is
how the identical code runs on Colab (where the roots are on mounted
Drive) and on a laptop.

RESEARCH_ROOT is still honoured as a legacy alias for DATA_ROOT's parent,
so older notebooks and shell sessions that set it keep working.
"""

import os
import sys

# The Windows console defaults to cp1252, which cannot encode the symbols
# these reports print (Δ, ±, →). Force UTF-8 so the same scripts run
# identically on Windows and on Colab.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# The repository root: two levels up from evaluation/rigor/paths.py.
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

# The dataset. Kept outside the repository on purpose -- see the module
# docstring. RESEARCH_ROOT is the legacy variable name and still works.
_legacy_root = os.environ.get("RESEARCH_ROOT")
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(_legacy_root, "archive") if _legacy_root
    else os.path.join(os.path.dirname(PROJECT_ROOT), "archive"),
)
# Retained so external_validation_isic.py and any older script that expects
# a RESEARCH_ROOT containing `archive/` continues to resolve correctly.
RESEARCH_ROOT = os.path.dirname(DATA_ROOT)

# --- inputs (produced by the experiment runs) -----------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, "experiments")

IMAGE_DIRS = [
    os.path.join(DATA_ROOT, "HAM10000_images_part_1"),
    os.path.join(DATA_ROOT, "HAM10000_images_part_2"),
]

# --- outputs (the rigor layer's evidence artifacts) -----------------------
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
RIGOR_DIR = os.path.join(ANALYSIS_DIR, "rigor")
PRED_DIR = os.path.join(RIGOR_DIR, "predictions")
FIG_DIR = os.path.join(RIGOR_DIR, "figures")
TABLE_DIR = os.path.join(RIGOR_DIR, "tables")

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
HIGH_RISK_CLASSES = {'mel', 'bcc', 'akiec'}
MODELS = ['efficientnet_b4', 'resnet50', 'densenet169']
METHODS = ['entropy', 'mc_dropout', 'margin', 'least_confidence']
POLICIES = ['dual_metric', 'uncertainty_only']
# Recent acquisition baselines, run as "<model>_baseline_<strategy>". They
# are parsed with policy='baseline' so that every existing analysis, which
# filters on policy in ('dual_metric', 'uncertainty_only'), continues to
# compare exactly what it did before and silently excludes them -- while
# the dedicated baseline comparison can select them by that same field.
BASELINES = ['coreset', 'badge', 'clue', 'vaal']
FINAL_ROUND = 15

# Consistent colours across every rigor figure.
COLOR_UNC = "#6b7280"   # gray  — uncertainty-only (baseline)
COLOR_DUAL = "#1b7a5e"  # teal  — dual-metric (ours)
COLOR_ACCENT = "#b45309"


def ensure_dirs():
    for d in [ANALYSIS_DIR, RIGOR_DIR, PRED_DIR, FIG_DIR, TABLE_DIR]:
        os.makedirs(d, exist_ok=True)


def parse_experiment_id(name):
    """
    'resnet50_entropy_dual_metric'  -> ('resnet50', 'entropy', 'dual_metric')
    'resnet50_baseline_badge'       -> ('resnet50', 'badge', 'baseline')
    'resnet50_entropy_dual_metric_s43'
                                    -> ('resnet50', 'entropy', 'dual_metric')

    Returns (None, None, None) for anything unrecognised; every caller
    treats that as "skip this directory", which is how stray folders are
    ignored.
    """
    # A non-baseline training seed appends "_s<seed>". Strip it so
    # multi-seed replicates parse as the configuration they replicate --
    # the seed itself is recorded inside results.csv and full.json.
    base = name
    if "_s" in base:
        head, _, tail = base.rpartition("_s")
        if tail.isdigit():
            base = head

    for model in sorted(MODELS, key=len, reverse=True):
        if not base.startswith(model):
            continue
        rest = base[len(model) + 1:]

        if rest.startswith("baseline_"):
            strategy = rest[len("baseline_"):]
            if strategy in BASELINES:
                return model, strategy, "baseline"
            continue

        for method in sorted(METHODS, key=len, reverse=True):
            if not rest.startswith(method):
                continue
            policy = rest[len(method) + 1:]
            if policy in POLICIES:
                return model, method, policy
    return None, None, None
