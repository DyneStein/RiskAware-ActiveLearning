# XAI — Explainability Pipeline

This folder contains the Explainable AI (XAI) analysis pipeline for the **RiskAware-ActiveLearning** research project. It generates and evaluates visual heatmap explanations for trained skin lesion classification models using three distinct methods.

---

## Folder Structure

```
XAI/
│
├── Data/               ← Place your input images here (.jpg / .png)
├── Model/              ← Place your trained model checkpoints here (.pt)
│
├── gradCam++/          ← Grad-CAM++ heatmap generation
│   ├── generate_gradcam_heatmaps.py
│   └── results/        ← Auto-created on first run
│
├── eigenCAM/           ← EigenCAM heatmap generation
│   ├── generate_eigencam_heatmaps.py
│   └── results/        ← Auto-created on first run
│
├── scoreCam/           ← Score-CAM heatmap generation
│   ├── generate_scorecam_heatmaps.py
│   └── results/        ← Auto-created on first run
│
└── README.md           ← This file
```

---

## What to Place Where

### `Data/` — Input Images
Drop any skin lesion images you want to explain here. Accepted formats: `.jpg`, `.jpeg`, `.png`.

The scripts will automatically detect and process **every image** in this folder. No configuration required.

> **Note:** The `Data/` folder contains a `.gitkeep` placeholder to preserve the folder in Git. Delete it or leave it — the scripts will ignore it.

### `Model/` — Trained Checkpoints
Drop your trained PyTorch model checkpoints (`.pt` files) here.

The scripts automatically detect all models in this folder and run the selected XAI method on every model–image pair.

> **Note:** Model files are **not** tracked by Git due to their large size (typically ~80–250 MB each). Store them locally or in a shared drive and add them to this folder before running.

#### Supported Architectures
The scripts **automatically infer the architecture** from the checkpoint's internal weight shapes. You do **not** need to rename your files. Supported:

| Architecture | Feature Dim |
| :--- | :---: |
| ResNet-50 | 2048 |
| DenseNet-169 | 1664 |
| EfficientNet-B4 | 1792 |

If none of the signatures match, the script defaults to EfficientNet-B4.

---

## How to Run

Make sure you have the required packages installed (see `requirements.txt` in the repository root).

Then, from inside any of the method folders, run:

```bash
# Grad-CAM++ (class-discriminative, gradient-based)
cd XAI/gradCam++
python generate_gradcam_heatmaps.py

# EigenCAM (gradient-free, PCA-based)
cd XAI/eigenCAM
python generate_eigencam_heatmaps.py

# Score-CAM (gradient-free, perturbation-based)
cd XAI/scoreCam
python generate_scorecam_heatmaps.py
```

On startup, each script will print:

```
Detected number of images: X
Detected number of models: Y
```

It then loops through every model, then every image, and prints a progress line like:

```
[1/2] Running Grad-CAM++ for Model: my_resnet_model
Architecture detected: resnet50 | Layer attached: Bottleneck
  [1/4] ISIC_0024306 -> Pred: MEL (97.3%)
  [2/4] ISIC_0025661 -> Pred: NV (88.1%)
--> Finished my_resnet_model in 43.2s!
```

---

## Output

Results are saved in a `results/` subfolder inside each method's directory, organized by model name:

```
gradCam++/results/
└── my_resnet_model/
    ├── ISIC_0024306_pred_mel_GradCAM++.png     ← Overlay visualization
    ├── ISIC_0024306_pred_mel_GradCAM++_raw.npy ← Raw heatmap array
    ├── ISIC_0025661_pred_nv_GradCAM++.png
    └── ISIC_0025661_pred_nv_GradCAM++_raw.npy
```

Each `.npy` file stores the normalized heatmap as a 2D float array (values in `[0, 1]`), which can be loaded for further quantitative analysis:

```python
import numpy as np
heatmap = np.load("ISIC_0024306_pred_mel_GradCAM++_raw.npy")
```

An `execution_log.txt` is also written inside each method's folder on every run, containing a full record of predictions and timings.



## Notes for Reproducibility

- All scripts use `seed_everything(42)` to fix random state across NumPy, Python, and PyTorch.
- All model weights are loaded with `strict=False`, so checkpoints with extra heads (e.g., the dual-metric risk head) load cleanly without errors.
- Scripts are designed to run on **CPU or GPU** automatically — no manual configuration needed.
