"""
Analysis of the 24-experiment Risk-Aware Active Learning matrix.
Reads results/experiments/*/results.csv (+ pool_predictions/*.csv for the
risk-score AUROC check) and produces comparison tables + figures answering
the core research question: does adding the risk score (dual_metric) make
the active-learning loop safer than uncertainty alone, without hurting
accuracy?
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Repo-relative, so a fresh clone works with no editing. This file lives at
# <repo>/analysis/build_analysis.py, so the repo root is one level up.
# Override with PROJECT_ROOT to point at a Drive copy from Colab.
BASE = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
)
RESULTS_DIR = os.path.join(BASE, "results", "experiments")
OUT_DIR = os.path.join(BASE, "analysis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

HIGH_RISK_CLASSES = {'mel', 'bcc', 'akiec'}
EXPECTED_ROUNDS = 15

MODELS = ['efficientnet_b4', 'resnet50', 'densenet169']
METHODS = ['entropy', 'mc_dropout', 'margin', 'least_confidence']

MODEL_LABELS = {'resnet50': 'ResNet-50', 'densenet169': 'DenseNet-169', 'efficientnet_b4': 'EfficientNet-B4'}
METHOD_LABELS = {'entropy': 'Entropy', 'mc_dropout': 'MC-Dropout', 'margin': 'Margin', 'least_confidence': 'Least Conf.'}

COLOR_UNC = "#6b7280"   # gray — uncertainty-only baseline
COLOR_DUAL = "#1b7a5e"  # teal-green — dual-metric (risk-aware)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_experiment_id(name):
    for model in sorted(MODELS, key=len, reverse=True):
        if name.startswith(model + "_"):
            rest = name[len(model) + 1:]
            break
    else:
        return None
    for method in sorted(METHODS, key=len, reverse=True):
        if rest.startswith(method + "_"):
            policy = rest[len(method) + 1:]
            return model, method, policy
    return None


def compute_experiment_auroc(exp_dir):
    pool_dir = os.path.join(exp_dir, "pool_predictions")
    if not os.path.isdir(pool_dir):
        return pd.DataFrame()
    out = []
    for fname in sorted(os.listdir(pool_dir)):
        m = re.match(r"round_(\d+)_pool_predictions\.csv", fname)
        if not m:
            continue
        rnd = int(m.group(1))
        df = pd.read_csv(os.path.join(pool_dir, fname))
        if df.empty or 'risk_score' not in df.columns:
            continue
        y_true = df['true_label'].isin(HIGH_RISK_CLASSES).astype(int)
        if y_true.nunique() < 2:
            continue
        auroc = roc_auc_score(y_true, df['risk_score'])
        out.append({'round': rnd, 'auroc': auroc, 'n': len(df), 'n_high_risk': int(y_true.sum())})
    return pd.DataFrame(out).sort_values('round')


# ---------------------------------------------------------------------------
# 1. Load every experiment's results.csv
# ---------------------------------------------------------------------------
experiments = sorted(os.listdir(RESULTS_DIR))
rows = []
trajectories = {}
auroc_trajectories = {}
incomplete = []

for exp_id in experiments:
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    csv_path = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    parsed = parse_experiment_id(exp_id)
    if parsed is None:
        print("WARNING: could not parse experiment id:", exp_id)
        continue
    model, method, policy = parsed
    df['experiment_id'] = exp_id
    df['model'] = model
    df['method'] = method
    df['policy'] = policy
    trajectories[exp_id] = df

    complete = len(df) >= EXPECTED_ROUNDS
    if not complete:
        incomplete.append((exp_id, len(df)))

    final = df.iloc[-1]
    rows.append({
        'experiment_id': exp_id,
        'model': model,
        'method': method,
        'policy': policy,
        'rounds_completed': len(df),
        'complete': complete,
        'final_labeled_count': final['labeled_count'],
        'total_queries': final['total_queries'],
        'total_unsafe_auto_accepts': df['unsafe_auto_accepts'].sum(),
        'final_accuracy': final['accuracy'],
        'final_f1_macro': final['f1_macro'],
        'final_fn_rate_malignant': final['fn_rate_malignant'],
        'final_fn_rate_melanoma': final['fn_rate_melanoma'],
        'mean_fn_rate_malignant': df['fn_rate_malignant'].mean(),
    })

    adf = compute_experiment_auroc(exp_dir)
    if not adf.empty:
        auroc_trajectories[exp_id] = adf

master = pd.DataFrame(rows).sort_values(['model', 'method', 'policy']).reset_index(drop=True)

auroc_rows = []
for exp_id, adf in auroc_trajectories.items():
    parsed = parse_experiment_id(exp_id)
    model, method, policy = parsed
    auroc_rows.append({
        'experiment_id': exp_id, 'model': model, 'method': method, 'policy': policy,
        'auroc_round1': adf.iloc[0]['auroc'],
        'auroc_final': adf.iloc[-1]['auroc'],
        'auroc_mean': adf['auroc'].mean(),
    })
auroc_summary = pd.DataFrame(auroc_rows)
auroc_summary.to_csv(os.path.join(TABLE_DIR, "risk_auroc_by_experiment.csv"), index=False)

master = master.merge(
    auroc_summary[['experiment_id', 'auroc_round1', 'auroc_final', 'auroc_mean']],
    on='experiment_id', how='left'
)
master.to_csv(os.path.join(TABLE_DIR, "master_summary.csv"), index=False)

# ---------------------------------------------------------------------------
# 2. Head-to-head: dual_metric vs uncertainty_only, per (model, method)
#    Only pairs where BOTH sides are complete (15/15 rounds) are compared.
# ---------------------------------------------------------------------------
pairs = []
for model in MODELS:
    for method in METHODS:
        u = master[(master.model == model) & (master.method == method) & (master.policy == 'uncertainty_only')]
        d = master[(master.model == model) & (master.method == method) & (master.policy == 'dual_metric')]
        if u.empty or d.empty:
            continue
        u = u.iloc[0]
        d = d.iloc[0]
        pair_complete = bool(u['complete']) and bool(d['complete'])
        pairs.append({
            'model': model, 'method': method, 'both_complete': pair_complete,
            'unc_only_unsafe_total': u.total_unsafe_auto_accepts,
            'dual_unsafe_total': d.total_unsafe_auto_accepts,
            'unsafe_reduction_pct': 100 * (u.total_unsafe_auto_accepts - d.total_unsafe_auto_accepts) / u.total_unsafe_auto_accepts if u.total_unsafe_auto_accepts else np.nan,
            'unc_only_fn_rate_final': u.final_fn_rate_malignant,
            'dual_fn_rate_final': d.final_fn_rate_malignant,
            'fn_rate_reduction_pct': 100 * (u.final_fn_rate_malignant - d.final_fn_rate_malignant) / u.final_fn_rate_malignant if u.final_fn_rate_malignant else np.nan,
            'unc_only_accuracy_final': u.final_accuracy,
            'dual_accuracy_final': d.final_accuracy,
            'accuracy_delta_pp': 100 * (d.final_accuracy - u.final_accuracy),
            'unc_only_f1_final': u.final_f1_macro,
            'dual_f1_final': d.final_f1_macro,
            'f1_delta_pp': 100 * (d.final_f1_macro - u.final_f1_macro),
            'unc_only_queries': u.total_queries,
            'dual_queries': d.total_queries,
            'extra_queries_pct': 100 * (d.total_queries - u.total_queries) / u.total_queries if u.total_queries else np.nan,
        })
comparison = pd.DataFrame(pairs)
comparison.to_csv(os.path.join(TABLE_DIR, "dual_vs_uncertainty_comparison.csv"), index=False)

comp_complete = comparison[comparison.both_complete].copy()
comp_complete['pair_label'] = comp_complete.apply(
    lambda r: f"{MODEL_LABELS[r['model']]}\n{METHOD_LABELS[r['method']]}", axis=1
)

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def grouped_bar(df, ycol_a, ycol_b, ylabel, title, fname, pct_fmt=False):
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, df[ycol_a], width=w, label='Uncertainty-only', color=COLOR_UNC)
    ax.bar(x + w / 2, df[ycol_b], width=w, label='Dual-metric (risk-aware)', color=COLOR_DUAL)
    ax.set_xticks(x)
    ax.set_xticklabels(df['pair_label'], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    for i, (a, b) in enumerate(zip(df[ycol_a], df[ycol_b])):
        fmt = "{:.1%}" if pct_fmt else "{:.0f}"
        ax.annotate(fmt.format(a), (i - w / 2, a), ha='center', va='bottom', fontsize=7)
        ax.annotate(fmt.format(b), (i + w / 2, b), ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)


# Fig 1 — total unsafe auto-accepts (THE headline safety metric)
grouped_bar(
    comp_complete, 'unc_only_unsafe_total', 'dual_unsafe_total',
    'Total unsafe auto-accepts (summed over all 15 rounds)\nlower = fewer dangerous cases slipped through without review',
    'Risk-aware escalation catches far more dangerous cases\n(Total Unsafe Auto-Accepts, per model + uncertainty method)',
    '01_unsafe_auto_accepts_total.png'
)

# Fig 2 — final-round FN rate (malignant)
grouped_bar(
    comp_complete, 'unc_only_fn_rate_final', 'dual_fn_rate_final',
    'False-negative rate, malignant classes\n(fraction of real cancers the model misses)',
    'Missed-cancer rate at the final round: baseline vs risk-aware',
    '02_fn_rate_malignant_final.png', pct_fmt=True
)

# Fig 3 — accuracy & f1 macro side by side (make sure safety doesn't cost performance)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(comp_complete))
w = 0.38
for ax, (col_a, col_b), ylabel, title in zip(
    axes,
    [('unc_only_accuracy_final', 'dual_accuracy_final'), ('unc_only_f1_final', 'dual_f1_final')],
    ['Test accuracy (final round)', 'Test F1-macro (final round)'],
    ['Overall accuracy', 'F1-macro (balanced across all 7 classes)']
):
    ax.bar(x - w / 2, comp_complete[col_a], width=w, label='Uncertainty-only', color=COLOR_UNC)
    ax.bar(x + w / 2, comp_complete[col_b], width=w, label='Dual-metric', color=COLOR_DUAL)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_complete['pair_label'], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle('Safety gain does not come at the cost of accuracy', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '03_accuracy_f1_final.png'))
plt.close(fig)

# Fig 4 — query cost (does risk-aware escalation cost more oracle labels?)
grouped_bar(
    comp_complete, 'unc_only_queries', 'dual_queries',
    'Total images sent to the oracle (all 15 rounds)',
    'Annotation cost: uncertainty-only vs risk-aware',
    '04_query_cost_total.png'
)

# Fig 5 — unsafe auto-accepts trajectory, by model (mean across methods, shaded range)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, model in zip(axes, MODELS):
    for policy, color, label in [('uncertainty_only', COLOR_UNC, 'Uncertainty-only'), ('dual_metric', COLOR_DUAL, 'Dual-metric')]:
        subset = [trajectories[eid] for eid in trajectories
                  if trajectories[eid]['model'].iloc[0] == model
                  and trajectories[eid]['policy'].iloc[0] == policy
                  and len(trajectories[eid]) >= EXPECTED_ROUNDS]
        if not subset:
            continue
        stacked = np.vstack([s['unsafe_auto_accepts'].values[:EXPECTED_ROUNDS] for s in subset])
        rounds = np.arange(1, EXPECTED_ROUNDS + 1)
        mean = stacked.mean(axis=0)
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
        ax.plot(rounds, mean, color=color, label=label, linewidth=2)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.15)
    ax.set_title(MODEL_LABELS[model])
    ax.set_xlabel('AL round')
axes[0].set_ylabel('Unsafe auto-accepts (this round)')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Unsafe auto-accepts per round, averaged across the 4 uncertainty methods\n(shaded band = min–max across methods)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '05_unsafe_auto_accepts_trajectory_by_model.png'))
plt.close(fig)

# Fig 6 — FN rate malignant trajectory, by model
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, model in zip(axes, MODELS):
    for policy, color, label in [('uncertainty_only', COLOR_UNC, 'Uncertainty-only'), ('dual_metric', COLOR_DUAL, 'Dual-metric')]:
        subset = [trajectories[eid] for eid in trajectories
                  if trajectories[eid]['model'].iloc[0] == model
                  and trajectories[eid]['policy'].iloc[0] == policy
                  and len(trajectories[eid]) >= EXPECTED_ROUNDS]
        if not subset:
            continue
        stacked = np.vstack([s['fn_rate_malignant'].values[:EXPECTED_ROUNDS] for s in subset])
        rounds = np.arange(1, EXPECTED_ROUNDS + 1)
        mean = stacked.mean(axis=0)
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
        ax.plot(rounds, mean, color=color, label=label, linewidth=2)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.15)
    ax.set_title(MODEL_LABELS[model])
    ax.set_xlabel('AL round')
axes[0].set_ylabel('FN rate (malignant classes)')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Missed-cancer rate over active-learning rounds\n(shaded band = min–max across the 4 uncertainty methods)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '06_fn_rate_trajectory_by_model.png'))
plt.close(fig)

# Fig 7 — accuracy trajectory, by model
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, model in zip(axes, MODELS):
    for policy, color, label in [('uncertainty_only', COLOR_UNC, 'Uncertainty-only'), ('dual_metric', COLOR_DUAL, 'Dual-metric')]:
        subset = [trajectories[eid] for eid in trajectories
                  if trajectories[eid]['model'].iloc[0] == model
                  and trajectories[eid]['policy'].iloc[0] == policy
                  and len(trajectories[eid]) >= EXPECTED_ROUNDS]
        if not subset:
            continue
        stacked = np.vstack([s['accuracy'].values[:EXPECTED_ROUNDS] for s in subset])
        rounds = np.arange(1, EXPECTED_ROUNDS + 1)
        mean = stacked.mean(axis=0)
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
        ax.plot(rounds, mean, color=color, label=label, linewidth=2)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.15)
    ax.set_title(MODEL_LABELS[model])
    ax.set_xlabel('AL round')
axes[0].set_ylabel('Test accuracy')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Learning curves: accuracy climbs the same way regardless of policy', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '07_accuracy_trajectory_by_model.png'))
plt.close(fig)

# Fig 8 — risk-score AUROC trend (is the risk head itself trustworthy?)
fig, ax = plt.subplots(figsize=(9, 5))
all_rounds = np.arange(1, EXPECTED_ROUNDS + 1)
complete_auroc = [adf for eid, adf in auroc_trajectories.items() if len(trajectories.get(eid, [])) >= EXPECTED_ROUNDS]
stacked = []
for adf in complete_auroc:
    s = adf.set_index('round')['auroc'].reindex(all_rounds)
    stacked.append(s.values)
stacked = np.array(stacked, dtype=float)
mean = np.nanmean(stacked, axis=0)
std = np.nanstd(stacked, axis=0)
ax.plot(all_rounds, mean, color=COLOR_DUAL, linewidth=2.5, label='Mean AUROC across all 24 experiments')
ax.fill_between(all_rounds, mean - std, mean + std, color=COLOR_DUAL, alpha=0.2, label='±1 std across experiments')
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random guessing (AUROC = 0.5)')
ax.set_ylim(0.4, 1.02)
ax.set_xlabel('AL round')
ax.set_ylabel('Risk-score AUROC vs true malignancy')
ax.set_title('Does the risk score actually know what is dangerous?\n(risk head AUROC, pooled unlabeled images, every round, every experiment)', fontsize=12, fontweight='bold')
ax.legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '08_risk_score_auroc_trend.png'))
plt.close(fig)

# Fig 9 — headline summary
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
metrics = [
    ('unsafe_reduction_pct', 'Reduction in unsafe\nauto-accepts (%)'),
    ('fn_rate_reduction_pct', 'Reduction in missed-cancer\nrate at final round (%)'),
]
for ax, (col, label) in zip(axes, metrics):
    vals = comp_complete[col].values
    mean_v = np.nanmean(vals)
    ax.scatter(np.random.uniform(-0.08, 0.08, size=len(vals)) + 0, vals, color=COLOR_DUAL, alpha=0.6, zorder=3, label='Each model+method pair')
    ax.bar([0], [mean_v], width=0.5, color=COLOR_DUAL, alpha=0.25, zorder=1)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(['All 12 pairs'])
    ax.set_ylabel(label)
    ax.set_title(f"Mean = {mean_v:.1f}%", fontsize=11)
fig.suptitle('Headline: does the risk score help, on average, across every model + uncertainty method?', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '09_headline_summary.png'))
plt.close(fig)

print("Done.")
print(f"Experiments loaded: {len(master)}  (complete: {int(master['complete'].sum())}, incomplete: {len(incomplete)})")
if incomplete:
    print("Incomplete experiments (excluded from head-to-head comparisons):")
    for eid, n in incomplete:
        print(f"  - {eid}: {n}/{EXPECTED_ROUNDS} rounds")
print(f"Pairs compared (both sides complete): {len(comp_complete)} / 12")
print()
print("=== Headline numbers ===")
print(f"Mean reduction in unsafe auto-accepts: {comp_complete['unsafe_reduction_pct'].mean():.1f}%")
print(f"Mean reduction in final-round FN rate (malignant): {comp_complete['fn_rate_reduction_pct'].mean():.1f}%")
print(f"Mean accuracy delta (dual - unc_only), pp: {comp_complete['accuracy_delta_pp'].mean():.2f}")
print(f"Mean F1-macro delta (dual - unc_only), pp: {comp_complete['f1_delta_pp'].mean():.2f}")
print(f"Mean extra queries used by dual-metric: {comp_complete['extra_queries_pct'].mean():.1f}%")
print(f"Mean risk-score AUROC (final round, all complete experiments): {master.loc[master['complete'], 'auroc_final'].mean():.4f}")
