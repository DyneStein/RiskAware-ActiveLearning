"""
Risk-Aware Active Learning Framework — Main Entry Point.

Usage:
  # Run a single experiment
  python main.py --model efficientnet_b4 --uncertainty entropy --policy dual_metric

  # Run all 24 experiments (3 models × 4 uncertainty × 2 policies)
  python main.py --run-all

  # Generate plots from saved results
  python main.py --plot-only
"""

import argparse
import os
import json
import copy
import torch
import numpy as np
import random

from config import (
    MODELS, UNCERTAINTY_METHODS, AL_ROUNDS,
    SEED_METADATA_CSV, POOL_METADATA_CSV, SEED_DATA_DIR,
    POOL_IMAGES_DIR, RESULTS_DIR, LOGS_DIR, PLOTS_DIR, RANDOM_SEED,
    RISK_THRESHOLD, USE_DYNAMIC_CLASS_WEIGHTS, QUERY_BUDGET_PER_ROUND,
    ensure_dirs,
)
from data.pool_manager import PoolManager
from active_learning.al_loop import run_experiment, build_experiment_id
from evaluation.visualization import generate_all_plots


def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_image_dirs():
    """
    Get all directories that contain images.
    Includes seed data and the remaining HAM10000 image directories.
    """
    dirs = [SEED_DATA_DIR]

    # Add HAM10000 image part directories
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        part_dir = os.path.join(POOL_IMAGES_DIR, part)
        if os.path.isdir(part_dir):
            dirs.append(part_dir)

    print(f"Image directories: {dirs}")
    return dirs


def run_single_experiment(model_name, uncertainty_method, policy_name,
                          num_rounds=AL_ROUNDS, resume=False,
                          risk_threshold=None, use_dynamic_weights=None,
                          query_budget=None):
    """Run a single experiment with fresh pool manager."""
    set_seed()
    ensure_dirs()
    image_dirs = get_image_dirs()

    # Fresh pool manager for each experiment
    pool_manager = PoolManager(
        seed_csv=SEED_METADATA_CSV,
        remaining_csv=POOL_METADATA_CSV,
    )

    results = run_experiment(
        model_name=model_name,
        uncertainty_method=uncertainty_method,
        policy_name=policy_name,
        pool_manager=pool_manager,
        image_dirs=image_dirs,
        num_rounds=num_rounds,
        resume=resume,
        risk_threshold=risk_threshold,
        use_dynamic_weights=use_dynamic_weights,
        query_budget=query_budget,
    )

    return results


def run_all_experiments(num_rounds=AL_ROUNDS, resume=False,
                        risk_threshold=None, use_dynamic_weights=None,
                        query_budget=None):
    """Run all 24 experiments (3 models × 4 uncertainty × 2 policies)."""
    policies = ['uncertainty_only', 'dual_metric']
    all_results = []

    total = len(MODELS) * len(UNCERTAINTY_METHODS) * len(policies)
    current = 0

    effective_dynamic_weights = (
        use_dynamic_weights if use_dynamic_weights is not None
        else USE_DYNAMIC_CLASS_WEIGHTS
    )

    for model in MODELS:
        for unc in UNCERTAINTY_METHODS:
            for policy in policies:
                current += 1
                experiment_id = build_experiment_id(
                    model, unc, policy, risk_threshold,
                    effective_dynamic_weights, query_budget,
                )

                # Skip experiments that are already fully complete
                done_path = os.path.join(LOGS_DIR, f"{experiment_id}_full.json")
                if resume and os.path.exists(done_path):
                    print(f"\n{'#'*70}")
                    print(f"# Experiment {current}/{total} — SKIPPING (already complete)")
                    print(f"#   {experiment_id}")
                    print(f"{'#'*70}")
                    with open(done_path, 'r') as f:
                        all_results.append(json.load(f))
                    continue

                print(f"\n{'#'*70}")
                print(f"# Experiment {current}/{total}")
                print(f"{'#'*70}")

                results = run_single_experiment(
                    model_name=model,
                    uncertainty_method=unc,
                    policy_name=policy,
                    num_rounds=num_rounds,
                    resume=resume,
                    risk_threshold=risk_threshold,
                    use_dynamic_weights=use_dynamic_weights,
                    query_budget=query_budget,
                )
                all_results.append(results)

    # Generate all plots
    generate_all_plots(all_results)

    # Save combined results
    combined_path = os.path.join(LOGS_DIR, 'all_experiments.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {combined_path}")

    return all_results


def load_and_plot():
    """Load saved results and regenerate plots."""
    combined_path = os.path.join(LOGS_DIR, 'all_experiments.json')
    if not os.path.exists(combined_path):
        # Try to load individual experiment results
        all_results = []
        for f in os.listdir(LOGS_DIR):
            if f.endswith('_full.json'):
                with open(os.path.join(LOGS_DIR, f), 'r') as fp:
                    all_results.append(json.load(fp))
        if not all_results:
            print("No saved results found. Run experiments first.")
            return
    else:
        with open(combined_path, 'r') as f:
            all_results = json.load(f)

    print(f"Loaded {len(all_results)} experiment results.")
    generate_all_plots(all_results)


def main():
    parser = argparse.ArgumentParser(
        description='Risk-Aware Active Learning Framework'
    )
    parser.add_argument('--model', type=str, default='efficientnet_b4',
                        choices=MODELS,
                        help='Model architecture')
    parser.add_argument('--uncertainty', type=str, default='entropy',
                        choices=UNCERTAINTY_METHODS,
                        help='Uncertainty estimation method')
    parser.add_argument('--policy', type=str, default='dual_metric',
                        choices=['uncertainty_only', 'dual_metric'],
                        help='Escalation policy')
    parser.add_argument('--rounds', type=int, default=AL_ROUNDS,
                        help='Number of active learning rounds')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all 24 experiments')
    parser.add_argument('--plot-only', action='store_true',
                        help='Generate plots from saved results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint (skip completed experiments in --run-all)')
    parser.add_argument('--risk-threshold', type=float, default=None,
                        help='MANUAL override for the clinical risk threshold '
                             '(dual-metric policy). If omitted (default), the '
                             'threshold is instead recalibrated automatically '
                             'every round (90th percentile on the current '
                             f'labeled set; config fallback: {RISK_THRESHOLD}). '
                             'Pass this for the threshold-sensitivity ablation sweep.')
    parser.add_argument('--use-dynamic-weights', action='store_true',
                        help='Train with inverse-frequency class weights '
                             '(recomputed from the labeled pool each round, '
                             'applied to both the classification and risk '
                             f'heads). Default: {USE_DYNAMIC_CLASS_WEIGHTS} '
                             '(config.USE_DYNAMIC_CLASS_WEIGHTS) — this flag '
                             'is a no-op unless the config default has been '
                             'changed to False.')
    parser.add_argument('--no-dynamic-weights', action='store_true',
                        help='Force OFF dynamic class weighting even though '
                             'it is on by default — use this to run the '
                             'unweighted ablation study deliberately.')
    parser.add_argument('--query-budget', type=int, default=None,
                        help='Top-K most-uncertain images escalated per '
                             'round, at minimum (0 = threshold-only, no '
                             f'budget floor). Default: {QUERY_BUDGET_PER_ROUND} '
                             '(config.QUERY_BUDGET_PER_ROUND). Any image '
                             'above the recalibrated threshold is still '
                             'escalated even past K — this only sets a '
                             'floor, not a ceiling.')

    args = parser.parse_args()

    if args.no_dynamic_weights:
        use_dynamic_weights = False
    elif args.use_dynamic_weights:
        use_dynamic_weights = True
    else:
        use_dynamic_weights = None  # falls back to config default

    if args.plot_only:
        load_and_plot()
    elif args.run_all:
        run_all_experiments(
            num_rounds=args.rounds,
            resume=args.resume,
            risk_threshold=args.risk_threshold,
            use_dynamic_weights=use_dynamic_weights,
            query_budget=args.query_budget,
        )
    else:
        run_single_experiment(
            model_name=args.model,
            uncertainty_method=args.uncertainty,
            policy_name=args.policy,
            num_rounds=args.rounds,
            resume=args.resume,
            risk_threshold=args.risk_threshold,
            use_dynamic_weights=use_dynamic_weights,
            query_budget=args.query_budget,
        )


if __name__ == '__main__':
    main()
