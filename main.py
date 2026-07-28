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
    POOL_IMAGES_DIR, RESULTS_DIR, EXPERIMENTS_DIR, LOGS_DIR, PLOTS_DIR,
    RANDOM_SEED, SPLIT_SEED, RISK_THRESHOLD, USE_DYNAMIC_CLASS_WEIGHTS,
    QUERY_BUDGET_PER_ROUND, ensure_dirs,
)
from data.pool_manager import PoolManager
from active_learning.al_loop import run_experiment, build_experiment_id
from evaluation.visualization import generate_all_plots
from active_learning.baselines import STRATEGIES as BASELINE_STRATEGIES
from tools.provenance import write_environment_record


def set_seed(seed=RANDOM_SEED):
    """
    Set the training seed for reproducibility.

    This covers weight initialisation, minibatch order, augmentation and
    dropout. It does NOT touch the train/test split, which is governed
    separately by config.SPLIT_SEED so that the held-out test set stays
    byte-identical no matter which training seed is used.

    The AL loop additionally re-seeds at the start of every round (see
    active_learning.al_loop.set_round_seed), so that a run resumed from a
    checkpoint reproduces an uninterrupted one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Training seed set to {seed} (test split fixed at SPLIT_SEED={SPLIT_SEED})")


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
                          query_budget=None, seed=None):
    """Run a single experiment with fresh pool manager."""
    effective_seed = seed if seed is not None else RANDOM_SEED
    set_seed(effective_seed)
    ensure_dirs()
    image_dirs = get_image_dirs()

    # Fresh pool manager for each experiment. random_seed is passed
    # EXPLICITLY as SPLIT_SEED, never as the training seed: the held-out
    # test set must stay byte-identical across every seed, otherwise
    # multi-seed runs are not comparable to one another or to the original
    # 24 experiments, and the image-level paired test loses its pairing.
    pool_manager = PoolManager(
        seed_csv=SEED_METADATA_CSV,
        remaining_csv=POOL_METADATA_CSV,
        random_seed=SPLIT_SEED,
    )

    experiment_id = build_experiment_id(
        model_name, uncertainty_method, policy_name, risk_threshold,
        use_dynamic_weights if use_dynamic_weights is not None
        else USE_DYNAMIC_CLASS_WEIGHTS,
        query_budget, effective_seed,
    )
    write_environment_record(
        os.path.join(EXPERIMENTS_DIR, experiment_id),
        experiment_id=experiment_id, seed=effective_seed,
        split_seed=SPLIT_SEED,
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
        seed=effective_seed,
    )

    return results


def run_baseline_experiment(model_name, strategy, num_rounds=AL_ROUNDS,
                            resume=False, query_budget=None, seed=None):
    """
    Run one baseline acquisition strategy (CoreSet / BADGE / CLUE / VAAL).

    The uncertainty method is pinned to 'entropy' because none of the four
    baselines uses one for selection — entropy is simply the cheapest
    scorer, and the pool is still scored every round so that the logged
    per-round CSVs have the same columns as every other experiment and
    remain readable by the whole rigor layer. The policy argument is
    likewise inert; `strategy` overrides step 4 entirely.
    """
    effective_seed = seed if seed is not None else RANDOM_SEED
    set_seed(effective_seed)
    ensure_dirs()
    image_dirs = get_image_dirs()

    pool_manager = PoolManager(
        seed_csv=SEED_METADATA_CSV,
        remaining_csv=POOL_METADATA_CSV,
        random_seed=SPLIT_SEED,
    )

    experiment_id = build_experiment_id(
        model_name, 'entropy', 'baseline', None,
        USE_DYNAMIC_CLASS_WEIGHTS, query_budget, effective_seed, strategy,
    )
    write_environment_record(
        os.path.join(EXPERIMENTS_DIR, experiment_id),
        experiment_id=experiment_id, seed=effective_seed,
        split_seed=SPLIT_SEED, strategy=strategy,
    )

    return run_experiment(
        model_name=model_name,
        uncertainty_method='entropy',
        policy_name='baseline',
        pool_manager=pool_manager,
        image_dirs=image_dirs,
        num_rounds=num_rounds,
        resume=resume,
        query_budget=query_budget,
        seed=effective_seed,
        strategy=strategy,
    )


def run_all_baselines(num_rounds=AL_ROUNDS, resume=False, seed=None,
                      models=None, strategies=None):
    """
    Run the full baseline comparison: 3 backbones x 4 strategies = 12 runs.

    Not 96. The 24-experiment matrix multiplies backbones by uncertainty
    methods by policies, but a baseline replaces the selection step
    entirely — it has no uncertainty dial and no policy dial, because it
    IS the policy. See active_learning/baselines/__init__.py.
    """
    models = models or MODELS
    strategies = strategies or list(BASELINE_STRATEGIES)
    effective_seed = seed if seed is not None else RANDOM_SEED

    all_results = []
    total = len(models) * len(strategies)
    current = 0

    for model in models:
        for strategy in strategies:
            current += 1
            experiment_id = build_experiment_id(
                model, 'entropy', 'baseline', None,
                USE_DYNAMIC_CLASS_WEIGHTS, None, effective_seed, strategy,
            )
            done_path = os.path.join(EXPERIMENTS_DIR, experiment_id, "full.json")
            if resume and os.path.exists(done_path):
                print(f"\n# Baseline {current}/{total} — SKIPPING (complete): "
                      f"{experiment_id}")
                with open(done_path, 'r') as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n{'#'*70}")
            print(f"# Baseline {current}/{total}: {experiment_id}")
            print(f"{'#'*70}")
            all_results.append(run_baseline_experiment(
                model_name=model, strategy=strategy, num_rounds=num_rounds,
                resume=resume, seed=effective_seed,
            ))

    combined_path = os.path.join(LOGS_DIR, 'all_baselines.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nBaseline results saved to: {combined_path}")
    return all_results


def run_all_experiments(num_rounds=AL_ROUNDS, resume=False,
                        risk_threshold=None, use_dynamic_weights=None,
                        query_budget=None, seed=None):
    """Run all 24 experiments (3 models × 4 uncertainty × 2 policies)."""
    policies = ['uncertainty_only', 'dual_metric']
    all_results = []
    effective_seed = seed if seed is not None else RANDOM_SEED

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
                    effective_dynamic_weights, query_budget, effective_seed,
                )

                # Skip experiments that are already fully complete
                done_path = os.path.join(EXPERIMENTS_DIR, experiment_id, "full.json")
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
                    seed=effective_seed,
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
        # Try to load individual experiment results from EXPERIMENTS_DIR
        all_results = []
        if os.path.isdir(EXPERIMENTS_DIR):
            for experiment_id in sorted(os.listdir(EXPERIMENTS_DIR)):
                full_json_path = os.path.join(EXPERIMENTS_DIR, experiment_id, 'full.json')
                if os.path.isfile(full_json_path):
                    with open(full_json_path, 'r') as fp:
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Training seed: weight initialisation, batch '
                             'order, augmentation, dropout. Default: '
                             f'{RANDOM_SEED} (config.RANDOM_SEED), which is '
                             'what all 24 original experiments used. Any '
                             'other value writes to a separate '
                             '"<experiment_id>_s<seed>" folder, so seeds can '
                             'never overwrite each other. The train/test '
                             'split is NOT affected — it is fixed by '
                             f'config.SPLIT_SEED={SPLIT_SEED} so every run '
                             'is evaluated on the identical test set.')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=list(BASELINE_STRATEGIES),
                        help='Run a recent active-learning BASELINE instead '
                             'of our escalation policy. Replaces the '
                             'selection step entirely; --uncertainty and '
                             '--policy are ignored. Each round it is given '
                             'exactly the number of labels dual-metric spent '
                             'in that round on the same backbone, so the '
                             'comparison is cost-matched — which requires '
                             '<model>_entropy_dual_metric to have finished '
                             'first.')
    parser.add_argument('--run-baselines', action='store_true',
                        help='Run the full baseline comparison: 3 backbones '
                             f'x {len(BASELINE_STRATEGIES)} strategies = '
                             f'{3 * len(BASELINE_STRATEGIES)} runs (NOT 96 — '
                             'baselines have no uncertainty or policy dial).')

    args = parser.parse_args()

    if args.no_dynamic_weights:
        use_dynamic_weights = False
    elif args.use_dynamic_weights:
        use_dynamic_weights = True
    else:
        use_dynamic_weights = None  # falls back to config default

    if args.plot_only:
        load_and_plot()
    elif args.run_baselines:
        run_all_baselines(
            num_rounds=args.rounds,
            resume=args.resume,
            seed=args.seed,
        )
    elif args.strategy is not None:
        run_baseline_experiment(
            model_name=args.model,
            strategy=args.strategy,
            num_rounds=args.rounds,
            resume=args.resume,
            query_budget=args.query_budget,
            seed=args.seed,
        )
    elif args.run_all:
        run_all_experiments(
            num_rounds=args.rounds,
            resume=args.resume,
            risk_threshold=args.risk_threshold,
            use_dynamic_weights=use_dynamic_weights,
            query_budget=args.query_budget,
            seed=args.seed,
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
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
