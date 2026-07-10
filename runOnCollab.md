# ☁️ Running on Google Colab

Running this project on Google Colab is highly recommended to take advantage of free GPUs (like the T4). Because the dataset is large (~10,000 images) and the MC Dropout experiments take time, the best approach is to store everything on Google Drive and run the code directly from there.

This guide ensures that if Colab disconnects, your progress is safely saved to Google Drive and can be instantly resumed.

---

## Step 1: Upload Files to Google Drive

First, you need your code and data in your Google Drive so Colab can access them.

1. Open Google Drive and create a folder, e.g., `RiskAware-AL-Research`.
2. Upload the entire `RiskAware-ActiveLearning` code folder into it.
3. Upload the dataset folder (`Ham-1000000 Dataset for skin`) into it.

Your Google Drive structure should look exactly like this:
```text
My Drive/
└── RiskAware-AL-Research/
    ├── Ham-1000000 Dataset for skin/    # The dataset folder
    │   ├── HAM10000_images_part_1/      # Folder containing ~5,000 images (ISIC_*.jpg)
    │   ├── HAM10000_images_part_2/      # Folder containing ~5,000 images (ISIC_*.jpg)
    │   └── HAM10000_metadata.csv        # The original metadata file
    │
    └── RiskAware-ActiveLearning/        # The code folder
        ├── active_learning/
        ├── config.py
        └── ...
```

---

## Step 2: Set Up the Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. **Enable the GPU:**
   - Go to `Runtime` > `Change runtime type` in the top menu.
   - Select **T4 GPU** (or any available GPU) under "Hardware accelerator".
   - Click Save.

---

## Step 3: Mount Google Drive

In the first cell of your notebook, paste this code to connect your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

*Run the cell and follow the prompts to grant Colab access to your Drive.*

---

## Step 4: Update Paths for Colab

Colab's file system is different from your local Windows machine. We need to tell the code where things are on your Google Drive. 

Add a new cell with this code. It will dynamically update `config.py` without you having to edit the file manually every time:

```python
import os

# Define the paths to your folders on Google Drive
project_path = '/content/drive/MyDrive/RiskAware-AL-Research/RiskAware-ActiveLearning'
dataset_path = '/content/drive/MyDrive/RiskAware-AL-Research/Ham-1000000 Dataset for skin'

# Change directory to the project folder
os.chdir(project_path)

# Update config.py dynamically for Colab
config_content = f"""
import os

PROJECT_ROOT = r"{project_path}"
SEED_DATA_DIR = os.path.join(PROJECT_ROOT, "Seed Data")
SEED_METADATA_CSV = os.path.join(SEED_DATA_DIR, "seed_metadata.csv")

POOL_IMAGES_DIR = r"{dataset_path}"
POOL_METADATA_CSV = os.path.join(PROJECT_ROOT, "Oracle_Simulated_Doctor", "MetaData of Dataset (not seed data).csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
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
QUERY_BUDGET_PER_ROUND = 0
SEED_PER_CLASS = 70
TEST_SPLIT_RATIO = 0.20

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS_PER_ROUND = 10
IMAGE_SIZE = 224
NUM_WORKERS = 2 
USE_DYNAMIC_CLASS_WEIGHTS = False

# Fallback only — actual thresholds are seed-calibrated per experiment,
# see active_learning/al_loop.py calibrate_thresholds()
UNCERTAINTY_THRESHOLD = 0.5
RISK_THRESHOLD = 0.3
RANDOM_SEED = 42

def ensure_dirs():
    for d in [RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, PLOTS_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
"""

with open("config.py", "w") as f:
    f.write(config_content)
    
print("config.py updated for Colab paths and working directory set to:", os.getcwd())
```

*Run this cell once. It ensures all results (checkpoints, logs, plots) are saved directly back to your Google Drive.*

---

## Step 5: Run Your Experiments!

Now you can run the terminal commands by prefixing them with an exclamation mark (`!`) in a new cell.

**To run a quick test (2 rounds):**
```python
!python main.py --model resnet50 --uncertainty entropy --policy dual_metric --rounds 2
```

**To run all 24 experiments:**
```python
# ALWAYS use --resume on Colab! If Colab kicks you off after a few hours, 
# you just run this exact cell again and it picks up right where it left off.
!python main.py --run-all --resume
```

**To run with a custom risk threshold:**
```python
# Lower threshold (0.2) = more aggressive safety (more images escalated to oracle)
!python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.2

# Higher threshold (0.5) = fewer oracle queries but potentially less safe
!python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold 0.5
```

**To run a full threshold sensitivity analysis:**
```python
# This is great for strengthening the paper — shows the safety vs. cost tradeoff
for rt in [0.1, 0.2, 0.3, 0.4, 0.5]:
    !python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --risk-threshold {rt} --resume
```

Each threshold value saves to a separate experiment (e.g., `..._rt0.1`, `..._rt0.2`), so results never overwrite each other.

**To run the dynamic class-weighting ablation** (inverse-frequency loss weights, recomputed from the labeled pool each round):
```python
!python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric --use-dynamic-weights
```
This saves to a separate experiment ID (suffix `_dynw`), so it never overwrites the unweighted baseline run.

> **Note on thresholds:** escalation thresholds are no longer hardcoded at 0.5/0.3 — each experiment calibrates its own thresholds automatically from the 490 seed images in round 1 (90th percentile). `--risk-threshold` still works as a manual override for the sweep above; there's no manual override for the uncertainty threshold. See `RESEARCHER.md` → "Seed-Calibrated Thresholds" for details.

---

## 💡 Pro-Tips for Colab

1. **The Magic of `--resume`:** Free Colab tiers disconnect if you close the tab or after a certain time limit. Because we built a robust checkpoint system, `--resume` is your best friend. Even if you haven't started an experiment yet, using `--resume` is safe (it will just start from round 1).
2. **Check your progress:** While the notebook is running, you can open your Google Drive in another tab, go to `RiskAware-AL-Research/RiskAware-ActiveLearning/results/logs/`, and view the `_results.csv` files to see the metrics live.
3. **Keep the tab open:** Colab will disconnect you if it thinks you are idle. Keep the browser tab open and visible while running long experiments.
4. **Preventing Disconnects:** Sometimes running this simple Javascript snippet in your browser's developer console (F12) can prevent Colab from timing out due to inactivity:
   ```javascript
   function ConnectButton(){
       console.log("Clicking Connect button"); 
       document.querySelector("colab-connect-button").click() 
   }
   setInterval(ConnectButton,60000);
   ```
