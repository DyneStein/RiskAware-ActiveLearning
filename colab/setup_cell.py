# =====================================================================
#  ONE-CELL COLAB SETUP  —  paste this whole block into a single cell
#  Run it every time you reconnect, after ANY disconnect.
#
#  Order: mount Drive -> unzip dataset -> sync code from GitHub ->
#         rewrite config.py -> set paths -> print environment
#
#  Nothing about your results depends on this cell. It rebuilds only the
#  ENVIRONMENT (code + a local copy of the dataset). Every completed
#  round, checkpoint and log lives on Drive and is untouched by it.
# =====================================================================

import os
import zipfile

from google.colab import drive

# --- 1. Mount Google Drive -------------------------------------------
# The first time in a fresh runtime this pops up a Google sign-in window.
# Click through and grant access — Google requires that interactively,
# once per runtime. If it is already mounted, this just confirms.
drive.mount('/content/drive')

# --- 2. Settings you might change ------------------------------------
# The training seed. 42 is the baseline every original experiment used.
# Change it ONLY for the multi-seed replication runs, and remember that
# results land in a separate "<experiment_id>_s<seed>" folder so nothing
# is ever overwritten. The test split does NOT move with this — that is
# SPLIT_SEED below, which is frozen forever.
TRAINING_SEED = 42

DRIVE_ROOT = '/content/drive/MyDrive/Research/Active_learning and HITL'
DATASET_ZIP = f'{DRIVE_ROOT}/Dataset/archive.zip'

# --- 3. Unzip the dataset onto Colab's local fast disk ----------------
# Reading 10,015 JPEGs per round straight from Drive is roughly an order
# of magnitude slower than from local disk. Skips automatically if
# already unpacked this session, so re-running this cell costs nothing.
extract_to = '/content/dataset'
dataset_path = os.path.join(extract_to, 'archive')
marker = os.path.join(dataset_path, 'HAM10000_images_part_1')

if os.path.exists(marker):
    print('Dataset already unpacked this session — skipping unzip.')
else:
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
        z.extractall(extract_to)
    print('Dataset unpacked:', os.listdir(extract_to))

# --- 4. Sync the code -------------------------------------------------
# Step 5 rewrites config.py every session, which leaves a local change to
# a tracked file; a plain `git pull` refuses to overwrite that and aborts.
# Nothing in this clone is meant to survive between sessions (config.py is
# regenerated below regardless), so force-sync to exactly match GitHub.
project_path = '/content/RiskAware-ActiveLearning'

if not os.path.exists(project_path):
    os.chdir('/content')
    !git clone https://github.com/DyneStein/RiskAware-ActiveLearning.git
else:
    os.chdir(project_path)
    !git fetch origin
    !git reset --hard origin/main

os.chdir(project_path)
print('Code ready at:', os.getcwd())

# --- 5. Rewrite config.py for this session ----------------------------
# NOTE: SPLIT_SEED must be present and must stay 42. It fixes which
# images form the held-out test set. RANDOM_SEED (the training seed)
# varies for multi-seed runs; SPLIT_SEED never does, so every run is
# evaluated on the byte-identical 1,905-image test set and results stay
# comparable to the original 24 experiments.
config_content = f'''
import os

PROJECT_ROOT = r"{project_path}"
SEED_DATA_DIR = os.path.join(PROJECT_ROOT, "Seed Data")
SEED_METADATA_CSV = os.path.join(SEED_DATA_DIR, "seed_metadata.csv")

POOL_IMAGES_DIR = r"{dataset_path}"
POOL_METADATA_CSV = os.path.join(PROJECT_ROOT, "Oracle_Simulated_Doctor", "MetaData of Dataset (not seed data).csv")

RESULTS_DIR = r"{DRIVE_ROOT}/results"
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, "experiments")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

NUM_CLASSES = 7
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {{name: idx for idx, name in enumerate(CLASS_NAMES)}}
IDX_TO_CLASS = {{idx: name for idx, name in enumerate(CLASS_NAMES)}}

MODELS = ['efficientnet_b4', 'resnet50', 'densenet169']
UNCERTAINTY_METHODS = ['entropy', 'mc_dropout', 'margin', 'least_confidence']
MC_DROPOUT_PASSES = 30

AL_ROUNDS = 15
QUERY_BUDGET_PER_ROUND = 150
SEED_PER_CLASS = 70
TEST_SPLIT_RATIO = 0.20

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS_PER_ROUND = 10
IMAGE_SIZE = 224
NUM_WORKERS = 2
USE_DYNAMIC_CLASS_WEIGHTS = True

UNCERTAINTY_THRESHOLD = 0.5
RISK_THRESHOLD = 0.3

# Frozen forever — decides the held-out test set. Never change this.
SPLIT_SEED = 42
# The training seed for this session. Varies for multi-seed replication.
RANDOM_SEED = {TRAINING_SEED}

def ensure_dirs():
    for d in [RESULTS_DIR, CHECKPOINTS_DIR, EXPERIMENTS_DIR, LOGS_DIR, PLOTS_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
'''

with open('config.py', 'w') as f:
    f.write(config_content)

# --- 6. Point the analysis layer at Drive too -------------------------
# The rigor scripts read results and write figures/tables. On Colab those
# belong on Drive so they survive the runtime being recycled.
os.environ['PROJECT_ROOT'] = DRIVE_ROOT
os.environ['DATA_ROOT'] = dataset_path

# --- 7. Show what we are actually running on --------------------------
# Colab hands out different GPUs on different days, and a T4 and an A100
# differ by roughly 4x. Worth knowing before starting a 3-hour run.
print('=' * 62)
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "No GPU! Runtime > Change runtime type > GPU"
import torch
print(f'torch {torch.__version__} | CUDA {torch.version.cuda} | '
      f'available={torch.cuda.is_available()}')
print(f'Training seed: {TRAINING_SEED}   Split seed: 42 (frozen)')
print(f'Results -> {DRIVE_ROOT}/results')
print('=' * 62)
print('SETUP COMPLETE — ready to run an experiment command.')
print('=' * 62)
