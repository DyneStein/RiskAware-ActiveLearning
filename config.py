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
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

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
AL_ROUNDS = 15              # Number of active learning rounds
QUERY_BUDGET_PER_ROUND = 0  # 0 = unlimited (query all that policy selects)
SEED_PER_CLASS = 70         # Already done: 490 total seed images

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

# Inverse-frequency class weighting (sklearn 'balanced' formula) for the
# training loss. Off by default so it can be run as an ablation study via
# the --use-dynamic-weights CLI flag. See active_learning/al_loop.py
# compute_class_weights().
USE_DYNAMIC_CLASS_WEIGHTS = False

# ---------------------------------------------------------------------------
# Escalation Thresholds
# ---------------------------------------------------------------------------
# These are now FALLBACK values only. The active thresholds are calibrated
# per-experiment: after the model trains on the 490 seed images in round 1,
# it scores the seed set with itself and takes the 90th-percentile score as
# the fixed deployment threshold for the rest of the run (see
# active_learning/al_loop.py calibrate_thresholds()). This replaces a single
# hardcoded 0.5/0.3 that made no sense once uncertainty scores were left in
# their raw, unnormalized scale (methods like entropy and mc_dropout no
# longer share [0,1] with margin/least_confidence).
#
# RISK_THRESHOLD is still used as the "no override supplied" default for the
# --risk-threshold CLI flag, which stays available for the threshold-
# sensitivity ablation sweep documented in RESEARCHER.md.
UNCERTAINTY_THRESHOLD = 0.5  # Fallback only — normally overridden by calibration
RISK_THRESHOLD = 0.3         # Fallback only — normally overridden by calibration

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, PLOTS_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
