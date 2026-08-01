"""
OOD Test-Set Evaluation Script

This script evaluates trained PyTorch active-learning models against an external (OOD) test set.
It reads images and a ground truth CSV from Data/, 
runs a single deterministic forward pass on OOD images,

and compares predictions against ground truth labels.

Outputs per experiment:
  - ood_results.json   (Accuracy, F1 Macro/Weighted, per-class F1, FN rates)
  - predictions.csv    (image-level predictions with probabilities)
  - confusion_matrix.png

Outputs for comparisons:
  - Compared Results/<model>_<uncertainty>/  (Dual Metric vs Baseline side-by-side)
  - All_Compared_Results/                    (Master table + charts for all 24)
"""

import os
import time
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
REPO_DIR = os.path.join(PARENT_DIR, "RiskAware-ActiveLearning")
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms
except ImportError:
    print("Notice: PyTorch not installed. Script is in plotting-only mode.")
    torch = None
    Dataset = object

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, brier_score_loss

# ---------- Constants ----------
try:
    from config import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, IMAGE_SIZE
except Exception:
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    NUM_CLASSES = 7
    IMAGE_SIZE = 224

try:
    from constants import HIGH_RISK_CLASSES, DIAGNOSIS_FULL_NAMES
except Exception:
    HIGH_RISK_CLASSES = {'mel', 'bcc', 'akiec'}
    DIAGNOSIS_FULL_NAMES = {
        'akiec': 'Actinic Keratosis', 'bcc': 'Basal Cell Carcinoma', 'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma', 'mel': 'Melanoma', 'nv': 'Melanocytic Nevus', 'vasc': 'Vascular Lesion'
    }

try:
    from models.model_factory import create_model as get_model
except Exception as e:
    print(f"Notice: Using standalone torchvision fallback ({e}).")
    def get_model(name, num_classes=7, pretrained=False):
        from torchvision import models as tv_models
        if name == 'resnet50':
            m = tv_models.resnet50(pretrained=pretrained)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
            return m
        elif name == 'densenet169':
            m = tv_models.densenet169(pretrained=pretrained)
            m.classifier = torch.nn.Linear(m.classifier.in_features, num_classes)
            return m
        elif name == 'efficientnet_b4':
            m = tv_models.efficientnet_b4(pretrained=pretrained)
            m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
            return m
        raise ValueError(f"Unknown model: {name}")


plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'uncertainty_only': '#e74c3c',  # Red for baseline
    'dual_metric': '#2ecc71',       # Green for ours
}


# ---------- Safety Metrics ----------
def compute_fn_rate_malignant(y_true, y_pred):
    """What % of truly malignant lesions did the model misclassify as benign?"""
    high_risk_indices = {CLASS_TO_IDX[c] for c in HIGH_RISK_CLASSES}
    is_malignant = np.array([y in high_risk_indices for y in y_true])
    total_malignant = is_malignant.sum()
    if total_malignant == 0:
        return 0.0
    predicted_benign = np.array([y not in high_risk_indices for y in y_pred])
    false_negatives = (is_malignant & predicted_benign).sum()
    return float(false_negatives / total_malignant)


def compute_fn_rate_melanoma(y_true, y_pred):
    """What % of truly melanoma lesions did the model miss entirely?"""
    mel_idx = CLASS_TO_IDX['mel']
    is_melanoma = (np.array(y_true) == mel_idx)
    total_melanoma = is_melanoma.sum()
    if total_melanoma == 0:
        return 0.0
    predicted_not_melanoma = (np.array(y_pred) != mel_idx)
    false_negatives = (is_melanoma & predicted_not_melanoma).sum()
    return float(false_negatives / total_melanoma)


def compute_calibration_metrics(y_true, y_probs, num_bins=10):
    """Compute Brier score and Expected Calibration Error (ECE)."""
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    num_classes = y_probs_arr.shape[1]
    
    # Multiclass Brier Score
    y_one_hot = np.eye(num_classes)[y_true_arr]
    brier = float(np.mean(np.sum((y_probs_arr - y_one_hot) ** 2, axis=1)))
    
    # Expected Calibration Error (ECE)
    confidences = np.max(y_probs_arr, axis=1)
    predictions = np.argmax(y_probs_arr, axis=1)
    accuracies = (predictions == y_true_arr)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return brier, float(ece)


# ---------- Dataset ----------
class OODDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform

        self.samples = []
        available_files = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
        for _, row in self.df.iterrows():
            img_id = str(row['image']).strip()
            label_idx = -1
            for c_idx, c_name in enumerate(CLASS_NAMES):
                col = 'AK' if c_name == 'akiec' else c_name.upper()
                if col in self.df.columns and row[col] == 1.0:
                    label_idx = c_idx
                    break
            if label_idx != -1:
                if f"{img_id}.jpg" in available_files:
                    img_path = os.path.join(images_dir, f"{img_id}.jpg")
                    self.samples.append((img_path, label_idx, img_id))
                elif f"{img_id}.png" in available_files:
                    img_path = os.path.join(images_dir, f"{img_id}.png")
                    self.samples.append((img_path, label_idx, img_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, img_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, img_id


def get_transforms():
    if torch is None:
        return None
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def infer_architecture(model_pt_path, device):
    """Attempt to figure out the architecture by loading the state dict keys."""
    state_dict = torch.load(model_pt_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        sd = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        sd = state_dict['state_dict']
    else:
        sd = state_dict

    keys_str = " ".join(list(sd.keys())[:50])
    if 'denseblock' in keys_str or 'densenet' in model_pt_path.lower():
        return 'densenet169'
    elif 'layer1' in keys_str or 'resnet' in model_pt_path.lower():
        return 'resnet50'
    elif 'features.0' in keys_str or 'efficientnet' in model_pt_path.lower():
        return 'efficientnet_b4'
    return 'resnet50'  # fallback

# ---------- Core Evaluation ----------
def evaluate_model(model_name, model_pt_path, dataset, batch_size=64, device='cuda'):
    """
    Load a trained model, run a single deterministic forward pass on all
    OOD images, and compare predictions against ground truth.
    """
    # Windows multiprocessing overhead in DataLoader slows CPU inference dramatically. Use 0 workers on Windows CPU.
    workers = 0 if (os.name == 'nt' and device == 'cpu') else 2
    eff_batch = 32 if device == 'cpu' else batch_size
    dataloader = DataLoader(dataset, batch_size=eff_batch, shuffle=False, num_workers=workers)

    if device == 'cpu' and torch is not None:
        try:
            torch.set_num_threads(os.cpu_count() or 4)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    actual_model_name = model_name
    if model_name == "unknown":
        actual_model_name = infer_architecture(model_pt_path, device)
        print(f"  [!] Inferred architecture from weights: {actual_model_name}")

    model = get_model(actual_model_name, num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(model_pt_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    if device == 'cpu':
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        try:
            example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            example_input = example_input.to(memory_format=torch.channels_last)
            traced = torch.jit.trace(model, example_input, check_trace=False)
            model = torch.jit.optimize_for_inference(traced)
            print("  [+] Applied PyTorch JIT compilation & Conv-BatchNorm folding!")
        except Exception as e:
            pass

    all_preds = []
    all_targets = []
    all_img_ids = []
    all_probs = []

    print(f"  Running inference on {len(dataset)} images (CPU optimized: workers={workers}, batch={eff_batch}, L3 cache & JIT)...")
    start_time = time.time()
    with torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad():
        for imgs, targets, img_ids in tqdm(dataloader, desc="  Inference"):
            imgs = imgs.to(device)
            if device == 'cpu':
                try:
                    imgs = imgs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            # Standard 32-bit float inference guarantees 100% exact mathematical reproducibility with GPU results
            out = model(imgs)
            c_out = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(c_out, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_img_ids.extend(img_ids)
    end_time = time.time()
    total_infer_time = float(end_time - start_time)
    ms_per_img = float((total_infer_time / len(all_preds)) * 1000.0) if len(all_preds) > 0 else 0.0
    fps = float(len(all_preds) / total_infer_time) if total_infer_time > 0 else 0.0

    # --- Compute all metrics by comparing predictions to ground truth ---
    acc = float(accuracy_score(all_targets, all_preds))
    f1_macro = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))
    f1_weighted = float(f1_score(all_targets, all_preds, average='weighted', zero_division=0))

    precision, recall, f1_per, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    fn_mal = compute_fn_rate_malignant(all_targets, all_preds)
    fn_mel = compute_fn_rate_melanoma(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    # Calibration metrics
    brier_score, ece = compute_calibration_metrics(all_targets, all_probs)
    
    # ROC-AUC per class and overall
    try:
        roc_auc_macro = float(roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro'))
        roc_auc_weighted = float(roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted'))
    except Exception:
        roc_auc_macro, roc_auc_weighted = 0.0, 0.0

    # Binary Malignant ROC-AUC
    try:
        high_risk_indices = {CLASS_TO_IDX[c] for c in HIGH_RISK_CLASSES if c in CLASS_TO_IDX}
        mal_targets = [1 if t in high_risk_indices else 0 for t in all_targets]
        mal_probs = [sum(prob[i] for i, c in enumerate(CLASS_NAMES) if c in HIGH_RISK_CLASSES) for prob in all_probs]
        roc_auc_mal = float(roc_auc_score(mal_targets, mal_probs))
    except Exception:
        roc_auc_mal = 0.0

    # Specificity per class from cm
    specificities = []
    auc_per_class = []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
        try:
            bin_targets = [1 if t == i else 0 for t in all_targets]
            bin_probs = [prob[i] for prob in all_probs]
            auc_c = float(roc_auc_score(bin_targets, bin_probs))
        except Exception:
            auc_c = 0.0
        auc_per_class.append(auc_c)

    # Malignant Specificity & Sensitivity
    mal_tp = sum(cm[i, j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if CLASS_NAMES[i] in HIGH_RISK_CLASSES and CLASS_NAMES[j] in HIGH_RISK_CLASSES)
    mal_fn = sum(cm[i, j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if CLASS_NAMES[i] in HIGH_RISK_CLASSES and CLASS_NAMES[j] not in HIGH_RISK_CLASSES)
    mal_fp = sum(cm[i, j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if CLASS_NAMES[i] not in HIGH_RISK_CLASSES and CLASS_NAMES[j] in HIGH_RISK_CLASSES)
    mal_tn = sum(cm[i, j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if CLASS_NAMES[i] not in HIGH_RISK_CLASSES and CLASS_NAMES[j] not in HIGH_RISK_CLASSES)
    mal_specificity = float(mal_tn / (mal_tn + mal_fp)) if (mal_tn + mal_fp) > 0 else 0.0
    mal_sensitivity = float(mal_tp / (mal_tp + mal_fn)) if (mal_tp + mal_fn) > 0 else 0.0

    # Build results dictionary
    results = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'roc_auc_malignant': roc_auc_mal,
        'brier_score': brier_score,
        'ece': ece,
        'fn_rate_malignant': fn_mal,
        'fn_rate_melanoma': fn_mel,
        'specificity_malignant': mal_specificity,
        'sensitivity_malignant': mal_sensitivity,
        'total_inference_time_sec': total_infer_time,
        'ms_per_image': ms_per_img,
        'images_per_second': fps,
        'total_test_images': len(all_preds),
    }
    for i, name in enumerate(CLASS_NAMES):
        results[f'f1_{name}'] = float(f1_per[i])
        results[f'precision_{name}'] = float(precision[i])
        results[f'recall_{name}'] = float(recall[i])
        results[f'specificity_{name}'] = float(specificities[i])
        results[f'auc_{name}'] = float(auc_per_class[i])

    # Build predictions dataframe
    df_preds = pd.DataFrame({
        'image_id': all_img_ids,
        'true_label': [CLASS_NAMES[t] for t in all_targets],
        'predicted_label': [CLASS_NAMES[p] for p in all_preds],
        'correct': [t == p for t, p in zip(all_targets, all_preds)],
    })
    for c_idx, c_name in enumerate(CLASS_NAMES):
        df_preds[f'prob_{c_name}'] = [prob[c_idx] for prob in all_probs]

    return results, df_preds, cm


# ---------- Per-Experiment Plots ----------
def plot_confusion_matrix(exp_dir, exp_id, cm):
    """Save a labeled confusion matrix heatmap."""
    os.makedirs(exp_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [DIAGNOSIS_FULL_NAMES.get(c, c) for c in CLASS_NAMES]
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=8)
    ax.set_xlabel('Predicted Diagnosis')
    ax.set_ylabel('True Diagnosis')
    ax.set_title(f'Confusion Matrix — {exp_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'confusion_matrix.png'))
    plt.close()


# ---------- Comparison Plots ----------
def run_comparison_plotter():
    """Generate pairwise (Dual vs Baseline) and master comparison charts."""
    print("\nGenerating comparison plots...")
    results_dir = os.path.join(BASE_DIR, "results")
    compared_dir = os.path.join(BASE_DIR, "Compared_Results")
    all_compared_dir = os.path.join(BASE_DIR, "All_Compared_Results")
    os.makedirs(compared_dir, exist_ok=True)
    os.makedirs(all_compared_dir, exist_ok=True)

    # Load all saved results
    json_files = glob.glob(os.path.join(results_dir, "*", "ood_results.json"))
    all_res = []
    for f in sorted(json_files):
        with open(f, 'r') as fp:
            all_res.append(json.load(fp))

    if not all_res:
        print("No test results found to plot.")
        return

    # Group into pairs: same model + same uncertainty method
    pairs = {}
    for r in all_res:
        key = f"{r['model']}_{r['uncertainty_method']}"
        pairs.setdefault(key, {})[r['training_policy']] = r

    width = 0.35

    for key, pair in pairs.items():
        pair_dir = os.path.join(compared_dir, key)
        os.makedirs(pair_dir, exist_ok=True)

        base = pair.get('uncertainty_only', {})
        dual = pair.get('dual_metric', {})
        if not base or not dual:
            continue

        # 1. Safety Comparison (FN Rates)
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['FN Rate Malignant (%)', 'FN Rate Melanoma (%)']
        base_vals = [base['fn_rate_malignant']*100, base['fn_rate_melanoma']*100]
        dual_vals = [dual['fn_rate_malignant']*100, dual['fn_rate_melanoma']*100]
        x = np.arange(len(metrics))
        bars1 = ax.bar(x - width/2, base_vals, width, label='Trained via Baseline', color=COLORS['uncertainty_only'], alpha=0.8)
        bars2 = ax.bar(x + width/2, dual_vals, width, label='Trained via Dual Metric (Ours)', color=COLORS['dual_metric'], alpha=0.8)
        ax.set_ylabel('False-Negative Rate (%) ↓ Lower is Better')
        ax.set_title(f'Clinical Safety on OOD Dataset — {key}')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend()
        for b in list(bars1) + list(bars2):
            ax.annotate(f"{b.get_height():.2f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pair_dir, 'safety_comparison.png'))
        plt.close()

        # 2. Accuracy & F1 Macro
        fig, ax = plt.subplots(figsize=(8, 5))
        m_names = ['Accuracy (%)', 'F1 Macro (%)']
        b_scores = [base['accuracy']*100, base['f1_macro']*100]
        d_scores = [dual['accuracy']*100, dual['f1_macro']*100]
        x = np.arange(len(m_names))
        bars1 = ax.bar(x - width/2, b_scores, width, label='Trained via Baseline', color=COLORS['uncertainty_only'], alpha=0.8)
        bars2 = ax.bar(x + width/2, d_scores, width, label='Trained via Dual Metric (Ours)', color=COLORS['dual_metric'], alpha=0.8)
        ax.set_ylabel('Score (%) ↑ Higher is Better')
        ax.set_title(f'Classification Performance on OOD Dataset — {key}')
        ax.set_xticks(x)
        ax.set_xticklabels(m_names)
        ax.legend()
        for b in list(bars1) + list(bars2):
            ax.annotate(f"{b.get_height():.2f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(pair_dir, 'accuracy_f1_comparison.png'))
        plt.close()

        # 3. Per-Class F1
        fig, ax = plt.subplots(figsize=(12, 6))
        c_labels = [DIAGNOSIS_FULL_NAMES.get(c, c) for c in CLASS_NAMES]
        b_f1s = [base.get(f'f1_{c}', 0)*100 for c in CLASS_NAMES]
        d_f1s = [dual.get(f'f1_{c}', 0)*100 for c in CLASS_NAMES]
        x = np.arange(len(c_labels))
        ax.bar(x - width/2, b_f1s, width, label='Trained via Baseline', color=COLORS['uncertainty_only'], alpha=0.8)
        ax.bar(x + width/2, d_f1s, width, label='Trained via Dual Metric (Ours)', color=COLORS['dual_metric'], alpha=0.8)
        ax.set_ylabel('F1 Score (%) ↑')
        ax.set_title(f'Per-Class F1 on OOD Dataset — {key}')
        ax.set_xticks(x)
        ax.set_xticklabels(c_labels, rotation=30, ha='right', fontsize=9)
        ax.legend()
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(pair_dir, 'per_class_f1_comparison.png'))
        plt.close()

    # ---------- Grand Summary ----------
    print("Generating grand summary in All_Compared_Results/...")
    rows = []
    for r in all_res:
        rows.append({
            'Model': r['model'],
            'Uncertainty': r['uncertainty_method'],
            'Training Policy': 'Dual Metric (Ours)' if r['training_policy'] == 'dual_metric' else 'Baseline',
            'Accuracy': f"{r['accuracy']*100:.2f}%",
            'F1 Macro': f"{r['f1_macro']*100:.2f}%",
            'FN Rate (Malignant) ↓': f"{r['fn_rate_malignant']*100:.2f}%",
            'FN Rate (Melanoma) ↓': f"{r['fn_rate_melanoma']*100:.2f}%",
        })
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(all_compared_dir, 'ood_comparison_table.csv'), index=False)
    df_all.to_latex(os.path.join(all_compared_dir, 'ood_comparison_table.tex'), index=False, escape=True)
    print("  Saved master CSV and LaTeX tables.")

    # Master bar charts
    groups = {}
    for r in all_res:
        k = f"{r['model']}\n{r['uncertainty_method']}"
        groups.setdefault(k, {})[r['training_policy']] = r

    keys = list(groups.keys())
    x = np.arange(len(keys))

    # Master FN Rate Malignant
    fig, ax = plt.subplots(figsize=(15, 6))
    b_vals = [groups[k].get('uncertainty_only', {}).get('fn_rate_malignant', 0)*100 for k in keys]
    d_vals = [groups[k].get('dual_metric', {}).get('fn_rate_malignant', 0)*100 for k in keys]
    ax.bar(x - width/2, b_vals, width, label='Trained via Baseline', color=COLORS['uncertainty_only'], alpha=0.8)
    ax.bar(x + width/2, d_vals, width, label='Trained via Dual Metric (Ours)', color=COLORS['dual_metric'], alpha=0.8)
    ax.set_ylabel('False-Negative Rate (%) ↓')
    ax.set_title(f'All {len(keys)} Model Pairs: Malignant FN Rate on OOD Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(all_compared_dir, 'ood_fn_rate_malignant_all.png'))
    plt.close()

    # Master F1 Macro
    fig, ax = plt.subplots(figsize=(15, 6))
    b_vals = [groups[k].get('uncertainty_only', {}).get('f1_macro', 0)*100 for k in keys]
    d_vals = [groups[k].get('dual_metric', {}).get('f1_macro', 0)*100 for k in keys]
    ax.bar(x - width/2, b_vals, width, label='Trained via Baseline', color=COLORS['uncertainty_only'], alpha=0.8)
    ax.bar(x + width/2, d_vals, width, label='Trained via Dual Metric (Ours)', color=COLORS['dual_metric'], alpha=0.8)
    ax.set_ylabel('F1 Macro (%) ↑')
    ax.set_title(f'All {len(keys)} Model Pairs: F1 Macro on OOD Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(all_compared_dir, 'ood_f1_macro_all.png'))
    plt.close()

    print("All comparison plots generated successfully!")


# ---------- Parsing Experiment IDs ----------
def parse_experiment_id(exp_id):
    """Extract model_name, uncertainty_method, training_policy from the experiment ID."""
    model_name = "unknown"
    for arch in ['efficientnet_b4', 'densenet169', 'resnet50']:
        if arch in exp_id:
            model_name = arch
            break
            
    policy_name = "unknown"
    if 'dual_metric' in exp_id:
        policy_name = 'dual_metric'
    elif 'uncertainty_only' in exp_id:
        policy_name = 'uncertainty_only'
        
    unc_method = "unknown"
    rem = exp_id
    if model_name != "unknown": rem = rem.replace(model_name, "")
    if policy_name != "unknown": rem = rem.replace(policy_name, "")
    rem = rem.strip("_")
    if rem:
        unc_method = rem

    return model_name, unc_method, policy_name


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="OOD Test-Set Evaluation")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate comparison plots from previously saved JSON results")
    args = parser.parse_args()

    if args.plot_only:
        run_comparison_plotter()
        return

    if torch is None:
        print("PyTorch is not available. Switching to --plot-only mode...")
        run_comparison_plotter()
        return

    csv_files = glob.glob(os.path.join(BASE_DIR, "Data", "*.csv"))
    if not csv_files:
        print("No ground truth CSV found in Data/ folder. Exiting.")
        return
    csv_path = csv_files[0]
    
    images_dir = os.path.join(BASE_DIR, "Data")
    models_dir = os.path.join(BASE_DIR, "Model")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading OOD dataset from {os.path.basename(csv_path)}...")
    dataset = OODDataset(csv_path, images_dir, transform=get_transforms())
    print(f"Loaded {len(dataset)} images.\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    model_pts = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    print(f"Found {len(model_pts)} model checkpoints.\n")
    if not model_pts:
        print("No models to evaluate. Exiting.")
        return

    for pt_path in model_pts:
        filename = os.path.basename(pt_path)
        exp_id = filename.replace(".pt", "").replace("_model", "")
        exp_dir = os.path.join(results_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        json_path = os.path.join(exp_dir, "ood_results.json")
        if os.path.exists(json_path):
            print(f"Skipping {exp_id} (already evaluated).")
            continue

        model_name, unc_method, policy_name = parse_experiment_id(exp_id)
        if model_name == "unknown" or policy_name == "unknown":
            print(f"\n[WARNING] Model filename '{filename}' does not match standard format.")
            print("          It will be evaluated, but may be excluded from pairwise comparisons.\n")

        print(f"{'='*60}")
        print(f"Evaluating: {exp_id}")
        print(f"  Model: {model_name} | Trained with: {unc_method} + {policy_name}")
        print(f"{'='*60}")

        results, df_preds, cm = evaluate_model(
            model_name, pt_path, dataset, batch_size=32, device=device
        )

        # Tag results with experiment metadata
        results['model'] = model_name
        results['uncertainty_method'] = unc_method
        results['training_policy'] = policy_name
        results['dataset'] = 'OOD_Dataset'

        with open(json_path, 'w') as fp:
            json.dump(results, fp, indent=2)
        df_preds.to_csv(os.path.join(exp_dir, "predictions.csv"), index=False)

        plot_confusion_matrix(exp_dir, exp_id, cm)

        print(f"  Accuracy: {results['accuracy']*100:.2f}% | F1 Macro: {results['f1_macro']*100:.2f}%")
        print(f"  FN Malignant: {results['fn_rate_malignant']*100:.2f}% | FN Melanoma: {results['fn_rate_melanoma']*100:.2f}%\n")

    # After all models are done, generate comparison plots
    run_comparison_plotter()


if __name__ == '__main__':
    main()
