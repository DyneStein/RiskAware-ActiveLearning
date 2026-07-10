"""
Publication-ready visualization for the Risk-Aware Active Learning paper.

Generates 7 key plots:
1. Accuracy / F1 vs Oracle Queries
2. False-Negative Rate over AL Rounds
3. Unsafe Auto-Accept Rate (bar chart)
4. Main Comparison Table (LaTeX-ready)
5. 2×2 Grid Scatter (Uncertainty vs Risk)
6. Per-Class F1 (grouped bar chart)
7. Annotation Efficiency Curve
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/server

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PLOTS_DIR, TABLES_DIR, CLASS_NAMES
from constants import DIAGNOSIS_FULL_NAMES, HIGH_RISK_CLASSES


# Publication-quality style
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

# Color scheme
COLORS = {
    'uncertainty_only': '#e74c3c',   # Red for baseline
    'dual_metric': '#2ecc71',        # Green for ours
}
MODEL_MARKERS = {
    'efficientnet_b4': 'o',
    'resnet50': 's',
    'densenet169': '^',
}


def plot_accuracy_vs_queries(all_results, save_dir=None):
    """
    Plot 1: Accuracy/F1 vs cumulative oracle queries.
    Shows our system achieves same/better accuracy with fewer queries.
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for result in all_results:
        rounds = pd.DataFrame(result['rounds'])
        policy = result['policy']
        model = result['model']
        color = COLORS.get(policy, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')
        label = f"{model} ({policy.replace('_', ' ')})"

        axes[0].plot(
            rounds['total_queries'], rounds['accuracy'],
            color=color, marker=marker, label=label,
            linewidth=2, markersize=6
        )
        axes[1].plot(
            rounds['total_queries'], rounds['f1_macro'],
            color=color, marker=marker, label=label,
            linewidth=2, markersize=6
        )

    axes[0].set_xlabel('Cumulative Oracle Queries')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Classification Accuracy vs Oracle Queries')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Cumulative Oracle Queries')
    axes[1].set_ylabel('F1 Score (Macro)')
    axes[1].set_title('F1 Score vs Oracle Queries')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'accuracy_vs_queries.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_fn_rate_over_rounds(all_results, save_dir=None):
    """
    Plot 2: False-Negative Rate on malignant classes over AL rounds.
    The primary safety metric — should be consistently lower for dual-metric.
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in all_results:
        rounds = pd.DataFrame(result['rounds'])
        policy = result['policy']
        model = result['model']
        unc = result['uncertainty_method']
        color = COLORS.get(policy, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')
        label = f"{model} / {unc} ({policy.replace('_', ' ')})"

        ax.plot(
            rounds['round'], rounds['fn_rate_malignant'],
            color=color, marker=marker, label=label,
            linewidth=2, markersize=6, alpha=0.8
        )

    ax.set_xlabel('Active Learning Round')
    ax.set_ylabel('False-Negative Rate (Malignant Classes)')
    ax.set_title('Safety: False-Negative Rate Over Active Learning Rounds')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    path = os.path.join(save_dir, 'fn_rate_over_rounds.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_unsafe_auto_accepts(all_results, save_dir=None):
    """
    Plot 3: Unsafe auto-accept count (bar chart).
    Shows how many high-risk cases each policy incorrectly auto-accepted.
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Aggregate total unsafe auto-accepts per experiment
    labels = []
    baseline_unsafe = []
    ours_unsafe = []

    # Group by model+uncertainty
    groups = {}
    for result in all_results:
        key = f"{result['model']}\n{result['uncertainty_method']}"
        rounds = pd.DataFrame(result['rounds'])
        total_unsafe = rounds['unsafe_auto_accepts'].sum()
        if result['policy'] == 'uncertainty_only':
            groups.setdefault(key, {})['baseline'] = total_unsafe
        else:
            groups.setdefault(key, {})['ours'] = total_unsafe

    keys = list(groups.keys())
    baseline_vals = [groups[k].get('baseline', 0) for k in keys]
    ours_vals = [groups[k].get('ours', 0) for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width,
                   label='Uncertainty Only (Baseline)',
                   color=COLORS['uncertainty_only'], alpha=0.8)
    bars2 = ax.bar(x + width/2, ours_vals, width,
                   label='Dual Metric (Ours)',
                   color=COLORS['dual_metric'], alpha=0.8)

    ax.set_xlabel('Model / Uncertainty Method')
    ax.set_ylabel('Total Unsafe Auto-Accepts')
    ax.set_title('Safety: Dangerous Cases Incorrectly Auto-Accepted')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=9)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'unsafe_auto_accepts.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def generate_comparison_table(all_results, save_dir=None):
    """
    Plot 4: Main comparison table (saved as CSV and LaTeX).
    The central results table for the paper.
    """
    if save_dir is None:
        save_dir = TABLES_DIR
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for result in all_results:
        rounds = pd.DataFrame(result['rounds'])
        final = rounds.iloc[-1]

        rows.append({
            'Model': result['model'],
            'Uncertainty': result['uncertainty_method'],
            'Policy': result['policy'],
            'Accuracy': f"{final['accuracy']:.4f}",
            'F1 Macro': f"{final['f1_macro']:.4f}",
            'FN Rate (Malignant)': f"{final['fn_rate_malignant']:.4f}",
            'FN Rate (Melanoma)': f"{final['fn_rate_melanoma']:.4f}",
            'Total Queries': int(final['total_queries']),
            'Unsafe Auto-Accepts': int(rounds['unsafe_auto_accepts'].sum()),
        })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = os.path.join(save_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save as LaTeX
    latex_path = os.path.join(save_dir, 'comparison_table.tex')
    df.to_latex(latex_path, index=False, escape=True)
    print(f"  Saved: {latex_path}")

    return df


def plot_uncertainty_vs_risk_scatter(
    uncertainty_scores, risk_scores, true_labels,
    unc_threshold, risk_threshold, save_dir=None, title_suffix=""
):
    """
    Plot 5: 2×2 grid scatter plot — uncertainty vs risk score.
    Color-coded by true diagnosis. Shows the 4 quadrants.
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by high-risk vs low-risk true labels
    colors = []
    for label_idx in true_labels:
        class_name = CLASS_NAMES[label_idx]
        if class_name in HIGH_RISK_CLASSES:
            colors.append('#e74c3c')  # Red for malignant
        else:
            colors.append('#3498db')  # Blue for benign

    ax.scatter(uncertainty_scores, risk_scores,
               c=colors, alpha=0.5, s=20, edgecolors='none')

    # Axis limits are DYNAMIC, not hardcoded to [0, 1]. Uncertainty scores
    # are now reported in each method's own raw scale (e.g. entropy can
    # reach ~1.95, mc_dropout is small and sits near 0) — a fixed [0, 1]
    # box would silently clip points off the plot or misplace the quadrant
    # labels once thresholds are seed-calibrated instead of fixed at 0.5/0.3.
    unc_arr = np.asarray(uncertainty_scores)
    risk_arr = np.asarray(risk_scores)
    unc_max = max(float(unc_arr.max()) if unc_arr.size else 1.0, unc_threshold) * 1.15
    risk_max = max(float(risk_arr.max()) if risk_arr.size else 1.0, risk_threshold) * 1.15
    unc_max = unc_max if unc_max > 0 else 1.0
    risk_max = risk_max if risk_max > 0 else 1.0

    # Draw threshold lines
    ax.axvline(x=unc_threshold, color='black', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Unc threshold ({unc_threshold:.3f})')
    ax.axhline(y=risk_threshold, color='black', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Risk threshold ({risk_threshold:.3f})')

    # Label quadrants
    ax.text(unc_threshold/2, risk_threshold/2, 'AUTO-ACCEPT\n(Safe & Efficient)',
            ha='center', va='center', fontsize=10, color='green', fontweight='bold')
    ax.text(unc_threshold/2, (risk_max+risk_threshold)/2, 'ESCALATE\n(Safety Override!)',
            ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    ax.text((unc_max+unc_threshold)/2, risk_threshold/2, 'AUTO-ACCEPT\n(Low Risk)',
            ha='center', va='center', fontsize=10, color='green', fontweight='bold')
    ax.text((unc_max+unc_threshold)/2, (risk_max+risk_threshold)/2, 'ESCALATE\n(Always)',
            ha='center', va='center', fontsize=10, color='red', fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='High-Risk (mel/bcc/akiec)'),
        Patch(facecolor='#3498db', label='Low-Risk (nv/bkl/df/vasc)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('Uncertainty Score (raw)')
    ax.set_ylabel('Clinical Risk Score')
    ax.set_title(f'2×2 Dual-Metric Decision Grid {title_suffix}')
    ax.set_xlim(0, unc_max)
    ax.set_ylim(0, risk_max)

    path = os.path.join(save_dir, f'uncertainty_vs_risk_scatter{title_suffix.replace(" ", "_")}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_f1(all_results, save_dir=None):
    """
    Plot 6: Per-class F1 score comparison (grouped bar chart).
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Use the best model's results for clarity
    # Group by policy, average across models/uncertainty methods
    baseline_f1s = {c: [] for c in CLASS_NAMES}
    ours_f1s = {c: [] for c in CLASS_NAMES}

    for result in all_results:
        rounds = pd.DataFrame(result['rounds'])
        final = rounds.iloc[-1]
        for c in CLASS_NAMES:
            f1_key = f'f1_{c}'
            if f1_key in final:
                if result['policy'] == 'uncertainty_only':
                    baseline_f1s[c].append(final[f1_key])
                else:
                    ours_f1s[c].append(final[f1_key])

    # Average
    classes = CLASS_NAMES
    baseline_means = [np.mean(baseline_f1s[c]) if baseline_f1s[c] else 0 for c in classes]
    ours_means = [np.mean(ours_f1s[c]) if ours_f1s[c] else 0 for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color bars by risk category
    bar_colors_baseline = [
        '#ff6b6b' if c in HIGH_RISK_CLASSES else '#74b9ff'
        for c in classes
    ]
    bar_colors_ours = [
        '#ff4757' if c in HIGH_RISK_CLASSES else '#0984e3'
        for c in classes
    ]

    ax.bar(x - width/2, baseline_means, width,
           label='Uncertainty Only (Baseline)',
           color=bar_colors_baseline, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, ours_means, width,
           label='Dual Metric (Ours)',
           color=bar_colors_ours, alpha=0.9, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Diagnostic Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score: Baseline vs Dual-Metric')
    ax.set_xticks(x)
    ax.set_xticklabels([DIAGNOSIS_FULL_NAMES.get(c, c) for c in classes],
                       rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_f1.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_annotation_efficiency(all_results, save_dir=None):
    """
    Plot 7: Annotation efficiency curve.
    X-axis: % of dataset labeled
    Y-axis: accuracy achieved
    Shows dual-metric reaches same accuracy with fewer labels.
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    total_pool_size = None

    for result in all_results:
        rounds = pd.DataFrame(result['rounds'])
        policy = result['policy']
        model = result['model']
        color = COLORS.get(policy, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')
        label = f"{model} ({policy.replace('_', ' ')})"

        if total_pool_size is None:
            total_pool_size = rounds.iloc[0]['labeled_count'] + \
                              rounds.iloc[0]['unlabeled_count']

        pct_labeled = (rounds['labeled_count'] / total_pool_size) * 100

        ax.plot(
            pct_labeled, rounds['accuracy'],
            color=color, marker=marker, label=label,
            linewidth=2, markersize=6
        )

    ax.set_xlabel('% of Dataset Labeled')
    ax.set_ylabel('Accuracy')
    ax.set_title('Annotation Efficiency: Accuracy vs Labeling Cost')
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'annotation_efficiency.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots(all_results, save_dir=None):
    """Generate all 7 publication plots."""
    print("\nGenerating publication plots...")
    plot_accuracy_vs_queries(all_results, save_dir)
    plot_fn_rate_over_rounds(all_results, save_dir)
    plot_unsafe_auto_accepts(all_results, save_dir)
    generate_comparison_table(all_results, save_dir)
    plot_per_class_f1(all_results, save_dir)
    plot_annotation_efficiency(all_results, save_dir)
    print("All plots generated!\n")
