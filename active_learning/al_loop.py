"""
Main Active Learning Loop.

This is the heart of the experiment. For each round:
1. Train the model on the current labeled set
2. Score ALL unlabeled images (uncertainty + risk)
3. Apply the escalation policy to decide which images to query
4. Query the oracle for selected images → add to labeled set
5. Evaluate on the fixed test set
6. Log all metrics
7. Save checkpoint

Both policies (uncertainty-only and dual-metric) run through the SAME loop.
The ONLY difference is step 3 — the escalation decision logic.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS_PER_ROUND,
    IMAGE_SIZE, MC_DROPOUT_PASSES, NUM_WORKERS,
    UNCERTAINTY_THRESHOLD, RISK_THRESHOLD,
    CHECKPOINTS_DIR, LOGS_DIR, PLOTS_DIR, CLASS_NAMES,
)
from data.dataset import HAM10000Dataset
from data.transforms import get_train_transforms, get_eval_transforms
from models.model_factory import create_model
from uncertainty.uncertainty_factory import get_uncertainty_method
from risk_score.clinical_risk import compute_risk_score
from escalation import uncertainty_only, dual_metric
from evaluation.metrics import compute_all_metrics
import glob


def _get_checkpoint_dir(experiment_id):
    """Return the checkpoint directory for an experiment."""
    return os.path.join(CHECKPOINTS_DIR, experiment_id)


def _find_latest_checkpoint(experiment_id):
    """
    Find the latest completed round checkpoint for an experiment.

    Returns
    -------
    (round_num, checkpoint_dir) or (None, None) if no checkpoint found.
    """
    ckpt_base = _get_checkpoint_dir(experiment_id)
    if not os.path.isdir(ckpt_base):
        return None, None

    # Look for round_N subdirectories with a valid meta.json
    max_round = None
    max_dir = None
    for entry in os.listdir(ckpt_base):
        subdir = os.path.join(ckpt_base, entry)
        meta_path = os.path.join(subdir, "meta.json")
        if os.path.isdir(subdir) and os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                rnd = meta.get('completed_round')
                if rnd is not None and (max_round is None or rnd > max_round):
                    max_round = rnd
                    max_dir = subdir
            except (json.JSONDecodeError, KeyError):
                continue

    return max_round, max_dir


def score_unlabeled_pool(model, pool_dataset, uncertainty_method_name):
    """
    Score all images in the unlabeled pool with uncertainty and risk.

    Returns
    -------
    image_ids : list of str
    uncertainty_scores : np.ndarray
    risk_scores : np.ndarray
    predictions : np.ndarray (predicted class indices)
    all_probabilities : np.ndarray (B, num_classes)
    """
    loader = DataLoader(
        pool_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    all_image_ids = []
    all_labels = []
    all_uncertainty = []
    all_risk = []
    all_predictions = []
    all_probs_list = []

    model.eval()
    model.to(model.device)

    for images, labels, image_ids in tqdm(loader, desc="Scoring pool"):
        # Get probabilities
        if uncertainty_method_name == 'mc_dropout':
            # MC Dropout: multiple forward passes
            mean_probs, variance, mc_all_probs = model.predict_with_mc_dropout(
                images, n_passes=MC_DROPOUT_PASSES
            )
            probs = mean_probs

            # Compute MC Dropout uncertainty
            unc_fn = get_uncertainty_method('mc_dropout')
            # mc_all_probs shape: (n_passes, B, num_classes)
            unc_scores = unc_fn(mc_all_probs)
        else:
            # Single forward pass methods
            probs = model.predict(images)
            unc_fn = get_uncertainty_method(uncertainty_method_name)
            unc_scores = unc_fn(probs)

        # Compute risk scores
        risk = compute_risk_score(probs)

        # Predicted classes
        preds = np.argmax(probs, axis=1)

        all_image_ids.extend(list(image_ids))
        all_labels.extend(labels.numpy())
        all_uncertainty.extend(
            unc_scores if isinstance(unc_scores, np.ndarray)
            else [unc_scores]
        )
        all_risk.extend(risk if isinstance(risk, np.ndarray) else [risk])
        all_predictions.extend(preds)
        all_probs_list.append(probs)

    return (
        all_image_ids,
        np.array(all_uncertainty),
        np.array(all_risk),
        np.array(all_predictions),
        np.concatenate(all_probs_list, axis=0),
        np.array(all_labels),
    )


def evaluate_on_test(model, test_dataset):
    """Evaluate the model on the fixed test set."""
    loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    for images, labels, _ in tqdm(loader, desc="Evaluating test set"):
        probs = model.predict(images)
        preds = np.argmax(probs, axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.concatenate(all_probs, axis=0),
    )


def run_experiment(
    model_name,
    uncertainty_method,
    policy_name,       # 'uncertainty_only' or 'dual_metric'
    pool_manager,
    image_dirs,
    num_rounds=15,
    experiment_id=None,
    resume=False,
    risk_threshold=None,
):
    """
    Run a full active learning experiment.

    Parameters
    ----------
    model_name : str
        One of: 'efficientnet_b4', 'resnet50', 'densenet169'.
    uncertainty_method : str
        One of: 'entropy', 'mc_dropout', 'margin', 'least_confidence'.
    policy_name : str
        'uncertainty_only' or 'dual_metric'.
    pool_manager : PoolManager
        Manages labeled/unlabeled/test splits.
    image_dirs : list of str
        Directories containing images.
    num_rounds : int
        Number of active learning rounds.
    experiment_id : str, optional
        Unique ID for this experiment (for logging/checkpoints).
    resume : bool
        If True, attempt to resume from the latest checkpoint.
    risk_threshold : float, optional
        Clinical risk threshold for the dual-metric policy.
        If None, uses the default from config (0.3).

    Returns
    -------
    results : dict
        Full experiment results with per-round metrics.
    """
    # Use config defaults if not provided
    if risk_threshold is None:
        risk_threshold = RISK_THRESHOLD

    if experiment_id is None:
        experiment_id = f"{model_name}_{uncertainty_method}_{policy_name}"
        # Append threshold to ID if non-default, so different runs don't collide
        if risk_threshold != RISK_THRESHOLD:
            experiment_id += f"_rt{risk_threshold}"

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"  Model: {model_name}")
    print(f"  Uncertainty: {uncertainty_method}")
    print(f"  Policy: {policy_name}")
    print(f"  Risk Threshold: {risk_threshold}")
    print(f"  Rounds: {num_rounds}")
    print(f"{'='*70}\n")

    # Initialize model
    model = create_model(model_name)
    model.to(model.device)

    # Results storage
    round_results = []
    start_round = 1

    # --- Resume from checkpoint if requested ---
    if resume:
        last_round, ckpt_dir = _find_latest_checkpoint(experiment_id)
        if last_round is not None and ckpt_dir is not None:
            print(f"  ⟳ Resuming from checkpoint: round {last_round}")

            # Load model weights
            model_path = os.path.join(ckpt_dir, "model.pt")
            model.load_checkpoint(model_path)

            # Load pool state (labeled/unlabeled/test splits)
            pool_state_dir = os.path.join(ckpt_dir, "pool_state")
            pool_manager.load_state(pool_state_dir)

            # Load accumulated round results
            meta_path = os.path.join(ckpt_dir, "meta.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            round_results = meta.get('round_results', [])

            start_round = last_round + 1
            print(f"  ⟳ Will continue from round {start_round}")
        else:
            print("  No checkpoint found — starting from scratch.")

    for round_num in range(start_round, num_rounds + 1):
        round_start = time.time()
        print(f"\n--- Round {round_num}/{num_rounds} ---")

        # 1. Get current labeled and unlabeled data
        labeled_df = pool_manager.get_labeled()
        unlabeled_df = pool_manager.get_unlabeled()
        test_df = pool_manager.get_test()

        if len(unlabeled_df) == 0:
            print("Unlabeled pool is empty. Stopping early.")
            break

        print(f"  Labeled: {len(labeled_df)} | "
              f"Unlabeled: {len(unlabeled_df)} | "
              f"Test: {len(test_df)}")

        # 2. Train model on current labeled set
        train_dataset = HAM10000Dataset(
            labeled_df, image_dirs, transform=get_train_transforms(IMAGE_SIZE)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        model.train_model(
            train_loader, epochs=EPOCHS_PER_ROUND,
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # 3. Score all unlabeled images
        pool_dataset = HAM10000Dataset(
            unlabeled_df, image_dirs, transform=get_eval_transforms(IMAGE_SIZE)
        )
        (pool_ids, unc_scores, risk_scores,
         pool_preds, pool_probs, pool_true_labels) = score_unlabeled_pool(
            model, pool_dataset, uncertainty_method
        )

        # 4. Apply escalation policy
        if policy_name == 'uncertainty_only':
            decisions, escalate_idx, auto_accept_idx = uncertainty_only.decide(
                unc_scores, UNCERTAINTY_THRESHOLD
            )
            categories = np.array(['n/a'] * len(decisions))
        elif policy_name == 'dual_metric':
            decisions, escalate_idx, auto_accept_idx, categories = \
                dual_metric.decide(
                    unc_scores, risk_scores,
                    UNCERTAINTY_THRESHOLD, risk_threshold
                )
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # Save raw predictions and scores for this round
        round_preds_df = pd.DataFrame({
            'image_id': pool_ids,
            'true_label': [CLASS_NAMES[y] for y in pool_true_labels],
            'predicted_label': [CLASS_NAMES[y] for y in pool_preds],
            'uncertainty_score': unc_scores,
            'risk_score': risk_scores,
            'decision': decisions,
            'category': categories
        })
        os.makedirs(os.path.join(LOGS_DIR, experiment_id), exist_ok=True)
        round_preds_df.to_csv(
            os.path.join(LOGS_DIR, experiment_id, f"round_{round_num}_pool_predictions.csv"),
            index=False
        )

        # Plot 2x2 scatter for first and last round
        if policy_name == 'dual_metric' and (round_num == 1 or round_num == num_rounds):
            from evaluation.visualization import plot_uncertainty_vs_risk_scatter
            plot_uncertainty_vs_risk_scatter(
                unc_scores, risk_scores, pool_true_labels,
                UNCERTAINTY_THRESHOLD, RISK_THRESHOLD,
                save_dir=os.path.join(PLOTS_DIR, experiment_id),
                title_suffix=f"(Round {round_num})"
            )

        # 5. Query the oracle for escalated images
        escalated_ids = [pool_ids[i] for i in escalate_idx]
        print(f"  Escalated: {len(escalated_ids)} | "
              f"Auto-accepted: {len(auto_accept_idx)}")

        if len(escalated_ids) > 0:
            pool_manager.move_to_labeled(escalated_ids)

        # 6. Count unsafe auto-accepts (high-risk images that were auto-accepted)
        # Look up true labels for auto-accepted images
        unsafe_count = 0
        if len(auto_accept_idx) > 0:
            auto_accept_ids = [pool_ids[i] for i in auto_accept_idx]
            auto_accept_true_labels = []
            for img_id in auto_accept_ids:
                row = unlabeled_df[unlabeled_df['image_id'] == img_id]
                if not row.empty:
                    auto_accept_true_labels.append(row.iloc[0]['dx'])

            from constants import HIGH_RISK_CLASSES
            unsafe_count = sum(
                1 for dx in auto_accept_true_labels if dx in HIGH_RISK_CLASSES
            )

        # 7. Evaluate on test set
        test_dataset = HAM10000Dataset(
            test_df, image_dirs, transform=get_eval_transforms(IMAGE_SIZE)
        )
        test_labels, test_preds, test_probs = evaluate_on_test(
            model, test_dataset
        )

        metrics = compute_all_metrics(
            test_labels, test_preds, test_probs, CLASS_NAMES
        )

        # 8. Log round results
        round_time = time.time() - round_start
        round_data = {
            'round': round_num,
            'labeled_count': len(pool_manager.get_labeled()),
            'unlabeled_count': len(pool_manager.get_unlabeled()),
            'queries_this_round': len(escalated_ids),
            'total_queries': len(pool_manager.get_labeled()) - 490,
            'auto_accepted_this_round': len(auto_accept_idx),
            'unsafe_auto_accepts': unsafe_count,
            'round_time_seconds': round_time,
            **metrics,
        }
        round_results.append(round_data)

        print(f"  Accuracy: {metrics['accuracy']:.4f} | "
              f"F1 Macro: {metrics['f1_macro']:.4f} | "
              f"FN Rate (malignant): {metrics['fn_rate_malignant']:.4f} | "
              f"Unsafe auto-accepts: {unsafe_count}")

        # 9. Save checkpoint (model + pool state + results metadata)
        round_ckpt_dir = os.path.join(
            _get_checkpoint_dir(experiment_id), f"round_{round_num}"
        )
        os.makedirs(round_ckpt_dir, exist_ok=True)

        # Save model weights
        model.save_checkpoint(os.path.join(round_ckpt_dir, "model.pt"))

        # Save pool state (labeled/unlabeled/test CSVs)
        pool_manager.save_state(os.path.join(round_ckpt_dir, "pool_state"))

        # Save metadata for resume (completed round + accumulated results)
        meta = {
            'experiment_id': experiment_id,
            'model_name': model_name,
            'uncertainty_method': uncertainty_method,
            'policy_name': policy_name,
            'completed_round': round_num,
            'num_rounds': num_rounds,
            'round_results': round_results,
        }
        with open(os.path.join(round_ckpt_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        # Clean up previous round's checkpoint to save disk space
        if round_num > 1:
            prev_ckpt_dir = os.path.join(
                _get_checkpoint_dir(experiment_id), f"round_{round_num - 1}"
            )
            if os.path.isdir(prev_ckpt_dir):
                import shutil
                shutil.rmtree(prev_ckpt_dir)

        # Save round results incrementally (as CSV for easy inspection)
        os.makedirs(LOGS_DIR, exist_ok=True)
        results_df = pd.DataFrame(round_results)
        results_df.to_csv(
            os.path.join(LOGS_DIR, f"{experiment_id}_results.csv"),
            index=False
        )

    # Final results
    results = {
        'experiment_id': experiment_id,
        'model': model_name,
        'uncertainty_method': uncertainty_method,
        'policy': policy_name,
        'rounds': round_results,
    }

    # Save full results as JSON
    with open(os.path.join(LOGS_DIR, f"{experiment_id}_full.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {experiment_id}")
    print(f"  Final Accuracy: {round_results[-1]['accuracy']:.4f}")
    print(f"  Final F1 Macro: {round_results[-1]['f1_macro']:.4f}")
    print(f"  Final FN Rate (malignant): "
          f"{round_results[-1]['fn_rate_malignant']:.4f}")
    print(f"  Total Oracle Queries: {round_results[-1]['total_queries']}")
    print(f"{'='*70}\n")

    return results
