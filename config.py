"""
Central configuration for the Risk-Aware Active Learning framework.
All hyperparameters, paths, and experiment settings are defined here.
Paths are relative — they get overridden in Colab to point to Google Drive.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative — override these in Colab notebook)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SEED_DATA_DIR = os.path.join(PROJECT_ROOT, "Seed Data")
SEED_METADATA_CSV = os.path.join(SEED_DATA_DIR, "seed_metadata.csv")

# Remaining images (the unlabeled pool + test set source)
POOL_IMAGES_DIR = os.path.join(PROJECT_ROOT, "..", "Ham-1000000 Dataset for skin")
POOL_METADATA_CSV = os.path.join(
    PROJECT_ROOT, "Oracle_Simulated_Doctor", "MetaData of Dataset (not seed data).csv"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
# One folder per experiment_id: results.csv, full.json, pool_predictions/, and
# that experiment's own plots (confusion matrix per round, learning curve,
# 2x2 scatter). Populated automatically as each experiment runs — no need to
# run --run-all or --plot-only first. See active_learning/al_loop.py.
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, "experiments")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")      # cross-experiment: all_experiments.json
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")    # cross-experiment: the 6 comparison plots
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")  # cross-experiment: comparison_table.csv/.tex

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
NUM_CLASSES = 7
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = ['efficientnet_b4', 'resnet50', 'densenet169']

# ---------------------------------------------------------------------------
# Uncertainty Methods
# ---------------------------------------------------------------------------
UNCERTAINTY_METHODS = ['entropy', 'mc_dropout', 'margin', 'least_confidence']
MC_DROPOUT_PASSES = 30  # Number of forward passes for MC Dropout

# ---------------------------------------------------------------------------
# Active Learning
# ---------------------------------------------------------------------------
AL_ROUNDS = 15                # Number of active learning rounds
QUERY_BUDGET_PER_ROUND = 150  # Top-K most-uncertain images escalated per round
                               # (0 = threshold-only, no budget floor). Any
                               # image above the round's recalibrated
                               # uncertainty threshold is escalated even past
                               # K -- see escalation/uncertainty_only.py and
                               # escalation/dual_metric.py.
SEED_PER_CLASS = 70           # Already done: 490 total seed images

# Data split: 80% pool (for AL querying), 20% test (fixed, never touched)
TEST_SPLIT_RATIO = 0.20

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS_PER_ROUND = 10
IMAGE_SIZE = 224
NUM_WORKERS = 2  # DataLoader workers (set to 0 on Windows if issues)

# Inverse-frequency class weighting (sklearn 'balanced' formula) for both
# the classification head and the risk head's training loss. On by default
# -- HAM10000 is severely imbalanced (nv ~67% of all images) and unweighted
# loss biases the model toward "benign", inflating the missed-cancer rate.
# Run the unweighted ablation deliberately via --no-dynamic-weights. See
# active_learning/al_loop.py compute_class_weights() /
# compute_risk_class_weights().
USE_DYNAMIC_CLASS_WEIGHTS = True

# ---------------------------------------------------------------------------
# Escalation Thresholds
# ---------------------------------------------------------------------------
# These are FALLBACK values only. The real thresholds are recalibrated every
# round: the model scores the current labeled set with itself and takes the
# 90th-percentile score as that round's threshold (see
# active_learning/al_loop.py calibrate_thresholds()). Recalibrating every
# round (instead of once, in round 1) keeps the bar meaningful as the model
# improves -- a threshold calibrated once against an early, weak model goes
# stale within a couple of rounds and escalation collapses to zero.
#
# The uncertainty threshold works alongside QUERY_BUDGET_PER_ROUND (top-K):
# escalate the K most-uncertain images, plus anything above threshold even
# past K. The risk threshold has no budget at all -- any image the
# independent risk head calls "above threshold" is always escalated,
# uncapped, so a dangerous case is never skipped just because the K budget
# is full that round.
#
# RISK_THRESHOLD is still used as the "no override supplied" default for the
# --risk-threshold CLI flag, which stays available for the threshold-
# sensitivity ablation sweep documented in HOW_TO_RUN.txt.
UNCERTAINTY_THRESHOLD = 0.5  # Fallback only — normally overridden by calibration
RISK_THRESHOLD = 0.3         # Fallback only — normally overridden by calibration

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# TWO seeds, with deliberately different jobs. Do not merge them.
#
# SPLIT_SEED decides WHICH images form the held-out test set. It is frozen
# permanently at 42 and must never be varied. Every experiment ever run --
# past, present, and every future seed -- evaluates on the byte-identical
# 1,905-image test set. That is what makes the image-level paired McNemar
# test in evaluation/rigor/statistics.py valid: it compares two policies
# image by image on the same patients. Varying it would silently make runs
# incomparable and would confound training noise with test-set difficulty
# (with only ~9 'df' images in the whole test set, a reshuffle moves
# per-class metrics for reasons unrelated to the method).
SPLIT_SEED = 42
# RANDOM_SEED controls everything else that is random inside a run: weight
# initialisation, minibatch order, augmentation draws, and dropout masks.
# This is the one to vary for multi-seed replication (--seed on the command
# line). Runs with a seed other than the baseline 42 get an "_s<seed>"
# suffix on their experiment_id so they can never overwrite each other --
# see active_learning/al_loop.py build_experiment_id().
RANDOM_SEED = 42


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [RESULTS_DIR, CHECKPOINTS_DIR, EXPERIMENTS_DIR, LOGS_DIR, PLOTS_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
