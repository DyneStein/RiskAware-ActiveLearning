"""
Main Active Learning Loop.

This is the heart of the experiment. For each round:
1. Train the model (both heads, jointly) on the current labeled set
2. Recalibrate the escalation thresholds against the current labeled set
3. Score ALL unlabeled images (uncertainty from the classification head,
   risk from the independent risk head)
4. Apply the escalation policy to decide which images to query
5. Query the oracle for selected images → add to labeled set
6. Evaluate on the fixed test set
7. Log all metrics
8. Save checkpoint

Both policies (uncertainty-only and dual-metric) run through the SAME loop.
The ONLY difference is step 4 — the escalation decision logic.
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
    RISK_THRESHOLD, USE_DYNAMIC_CLASS_WEIGHTS, QUERY_BUDGET_PER_ROUND,
    CHECKPOINTS_DIR, EXPERIMENTS_DIR, LOGS_DIR, PLOTS_DIR, CLASS_NAMES,
)
from constants import HIGH_RISK_CLASSES
from data.dataset import HAM10000Dataset
from data.transforms import get_train_transforms, get_eval_transforms
from models.model_factory import create_model
from uncertainty.uncertainty_factory import get_uncertainty_method
from escalation import uncertainty_only, dual_metric
from evaluation.metrics import compute_all_metrics
from evaluation.visualization import (
    plot_confusion_matrix, plot_experiment_learning_curve,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import glob


def build_experiment_id(model_name, uncertainty_method, policy_name,
                         manual_risk_threshold=None, use_dynamic_weights=True,
                         query_budget=None):
    """
    Build the canonical experiment_id string, including suffixes for any
    non-default settings so different configurations never collide on disk.
    Used consistently by both main.py (resume-skip checks) and
    run_experiment() (actual run), so they always agree on the filename.

    Weighted training is the default (see config.USE_DYNAMIC_CLASS_WEIGHTS),
    so the suffix marks the unweighted ablation case, not the default.
    """
    experiment_id = f"{model_name}_{uncertainty_method}_{policy_name}"
    if manual_risk_threshold is not None and manual_risk_threshold != RISK_THRESHOLD:
        experiment_id += f"_rt{manual_risk_threshold}"
    if query_budget is not None and query_budget != QUERY_BUDGET_PER_ROUND:
        experiment_id += f"_k{query_budget}"
    if not use_dynamic_weights:
        experiment_id += "_noweights"
    return experiment_id


def compute_class_weights(labeled_df):
    """
    Inverse-frequency class weights from the CURRENT labeled set, using the
    same 'balanced' formula as sklearn.utils.class_weight.compute_class_weight:

        weight[c] = n_samples / (n_classes * count[c])

    A class with few labeled examples gets a bigger weight so the loss
    penalizes mistakes on it more — this is what actually applies the class
    weighting that config.USE_DYNAMIC_CLASS_WEIGHTS toggles on, for the
    7-class classification head.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Must have a 'dx' column with class codes matching CLASS_NAMES.

    Returns
    -------
    np.ndarray, shape (num_classes,)
        Weight for each class, in CLASS_NAMES order.
    """
    value_counts = labeled_df['dx'].value_counts()
    counts = np.array(
        [value_counts.get(name, 0) for name in CLASS_NAMES], dtype=np.float64
    )
    # Guard against a class with zero labeled examples (shouldn't happen —
    # the seed set has 70/class — but avoids a divide-by-zero if it ever does).
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return weights


def compute_risk_class_weights(labeled_df):
    """
    Inverse-frequency class weights for the binary risk head
    (non-malignant vs. malignant), same 'balanced' formula as
    compute_class_weights() but over two classes.

    Malignant cases (mel/bcc/akiec) are still the minority even in this
    binary framing, so unweighted training would still bias the risk head
    toward "predict non-malignant" — the same imbalance problem class
    weighting fixes for the classification head, one level down.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Must have a 'dx' column with class codes matching CLASS_NAMES.

    Returns
    -------
    np.ndarray, shape (2,)
        [weight for non-malignant, weight for malignant].
    """
    is_malignant = labeled_df['dx'].isin(HIGH_RISK_CLASSES)
    counts = np.array(
        [(~is_malignant).sum(), is_malignant.sum()], dtype=np.float64
    )
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (2 * counts)
    return weights


def calibrate_thresholds(model, calibration_df, image_dirs,
                          uncertainty_method_name, percentile=90):
    """
    Recalibrated escalation thresholds — run every round (not just once).

    Scores the current labeled set with the model that was JUST trained on
    it this round, then takes the 90th-percentile uncertainty score and
    90th-percentile risk score as this round's thresholds. Recalibrating
    every round (instead of locking in a value from round 1) keeps the bar
    meaningful as the model improves — a threshold calibrated once against
    an early, weak model goes stale within a couple of rounds and
    escalation collapses to zero, since a much-improved model rarely
    produces scores anywhere near what an early, confused version of
    itself used to.

    Because uncertainty scores are not forced into a shared [0, 1] range
    (see uncertainty/entropy.py, uncertainty/mc_dropout.py), a single
    threshold constant across methods wouldn't make sense — this function
    is what makes each method's threshold scale-appropriate automatically.

    Parameters
    ----------
    model : BaseModel
        The model, already trained on the current labeled set this round.
    calibration_df : pd.DataFrame
        The current labeled set (grows every round as images are escalated).
    image_dirs : list of str
        Directories containing images.
    uncertainty_method_name : str
        One of 'entropy', 'mc_dropout', 'margin', 'least_confidence'.
    percentile : float
        Percentile used as the threshold (default: 90th).

    Returns
    -------
    unc_threshold : float
    risk_threshold : float
    """
    calibration_dataset = HAM10000Dataset(
        calibration_df, image_dirs, transform=get_eval_transforms(IMAGE_SIZE)
    )
    (_, cal_unc, cal_risk, _, _, _) = score_unlabeled_pool(
        model, calibration_dataset, uncertainty_method_name
    )
    unc_threshold = float(np.percentile(cal_unc, percentile))
    risk_threshold = float(np.percentile(cal_risk, percentile))
    return unc_threshold, risk_threshold


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
    Score all images in the unlabeled pool with uncertainty (classification
    head) and risk (independent risk head).

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
        # Get classification probabilities (+ uncertainty)
        if uncertainty_method_name == 'mc_dropout':
            # MC Dropout: multiple forward passes through the classification
            # head only (the risk head is deterministic, scored separately).
            mean_probs, variance, mc_all_probs = model.predict_with_mc_dropout(
                images, n_passes=MC_DROPOUT_PASSES
            )
            probs = mean_probs

            unc_fn = get_uncertainty_method('mc_dropout')
            unc_scores = unc_fn(mc_all_probs)
        else:
            probs = model.predict(images)
            unc_fn = get_uncertainty_method(uncertainty_method_name)
            unc_scores = unc_fn(probs)

        # Risk score: one deterministic pass through the independent risk
        # head — not derived from the classification probabilities.
        risk = model.predict_risk(images)

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
    use_dynamic_weights=None,
    query_budget=None,
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
        MANUAL override for the clinical risk threshold (dual-metric policy
        only). If None (default), the risk threshold is instead
        recalibrated every round (see calibrate_thresholds()). Pass a
        value here to run the threshold-sensitivity ablation sweep instead
        of calibration.
    use_dynamic_weights : bool, optional
        If True (default via config), train both heads with inverse-
        frequency class weights recomputed from the labeled pool each
        round (see compute_class_weights() / compute_risk_class_weights()).
        Pass False to run the unweighted ablation. If None, uses
        config.USE_DYNAMIC_CLASS_WEIGHTS.
    query_budget : int, optional
        Top-K most-uncertain images escalated per round, at minimum (see
        escalation/uncertainty_only.py, escalation/dual_metric.py). If
        None, uses config.QUERY_BUDGET_PER_ROUND.

    Returns
    -------
    results : dict
        Full experiment results with per-round metrics.
    """
    manual_risk_threshold = risk_threshold  # None unless caller explicitly set it
    if use_dynamic_weights is None:
        use_dynamic_weights = USE_DYNAMIC_CLASS_WEIGHTS
    effective_query_budget = (
        query_budget if query_budget is not None else QUERY_BUDGET_PER_ROUND
    )

    # Everything specific to this experiment (results, raw predictions, and
    # this experiment's own plots) lives in one folder, populated
    # automatically as it runs -- no need to run --run-all or --plot-only
    # first. See config.EXPERIMENTS_DIR.
    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    experiment_plots_dir = os.path.join(experiment_dir, "plots")
    pool_predictions_dir = os.path.join(experiment_dir, "pool_predictions")

    if experiment_id is None:
        experiment_id = build_experiment_id(
            model_name, uncertainty_method, policy_name,
            manual_risk_threshold, use_dynamic_weights, query_budget,
        )

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"  Model: {model_name}")
    print(f"  Uncertainty: {uncertainty_method}")
    print(f"  Policy: {policy_name}")
    print(f"  Uncertainty Threshold: recalibrated every round (90th percentile)")
    print(f"  Query Budget (uncertainty, floor not ceiling): {effective_query_budget}/round")
    print(f"  Risk Threshold: "
          f"{'MANUAL override = ' + str(manual_risk_threshold) if manual_risk_threshold is not None else 'recalibrated every round (90th percentile), uncapped'}")
    print(f"  Dynamic Class Weights: {use_dynamic_weights}")
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

            # Load model weights (both heads + backbone)
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
            print(f"  ⟳ Thresholds recalibrate fresh this round, as every round")
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

        # 2. Train both heads jointly on current labeled set
        train_dataset = HAM10000Dataset(
            labeled_df, image_dirs, transform=get_train_transforms(IMAGE_SIZE)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        class_weights = (
            compute_class_weights(labeled_df) if use_dynamic_weights else None
        )
        risk_class_weights = (
            compute_risk_class_weights(labeled_df) if use_dynamic_weights else None
        )
        model.train_model(
            train_loader, epochs=EPOCHS_PER_ROUND,
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            class_weights=class_weights,
            risk_class_weights=risk_class_weights,
        )

        # 2b. Recalibrate escalation thresholds this round, using the
        # current (now-larger, now-smarter) labeled set. Recalculated
        # every round rather than locked in once, so the bar always
        # reflects the model's current behavior instead of going stale
        # as it improves round over round.
        calibrated_unc_threshold, calibrated_risk_threshold = \
            calibrate_thresholds(model, labeled_df, image_dirs, uncertainty_method)
        print(f"  ⚖ Recalibrated thresholds (90th percentile): "
              f"uncertainty={calibrated_unc_threshold:.4f}, "
              f"risk={calibrated_risk_threshold:.4f}")

        # The risk threshold actually used this round: manual CLI override
        # if one was given (for the threshold-sensitivity ablation sweep),
        # otherwise this round's recalibrated value.
        effective_unc_threshold = calibrated_unc_threshold
        effective_risk_threshold = (
            manual_risk_threshold if manual_risk_threshold is not None
            else calibrated_risk_threshold
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
                unc_scores, effective_unc_threshold, effective_query_budget
            )
            categories = np.array(['n/a'] * len(decisions))
        elif policy_name == 'dual_metric':
            decisions, escalate_idx, auto_accept_idx, categories = \
                dual_metric.decide(
                    unc_scores, risk_scores,
                    effective_unc_threshold, effective_risk_threshold,
                    effective_query_budget
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
        os.makedirs(pool_predictions_dir, exist_ok=True)
        round_preds_df.to_csv(
            os.path.join(pool_predictions_dir, f"round_{round_num}_pool_predictions.csv"),
            index=False
        )

        # Plot 2x2 scatter for first and last round
        if policy_name == 'dual_metric' and (round_num == 1 or round_num == num_rounds):
            from evaluation.visualization import plot_uncertainty_vs_risk_scatter
            plot_uncertainty_vs_risk_scatter(
                unc_scores, risk_scores, pool_true_labels,
                effective_unc_threshold, effective_risk_threshold,
                save_dir=experiment_plots_dir,
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

        # Confusion matrix — plotted every round into this experiment's own
        # folder (not persisted into round_data/results.csv, which would
        # clutter it with a stringified nested list; the plot is the record).
        cm = sk_confusion_matrix(
            test_labels, test_preds, labels=list(range(len(CLASS_NAMES)))
        )
        plot_confusion_matrix(
            cm, CLASS_NAMES, save_dir=experiment_plots_dir,
            title_suffix=f" (Round {round_num})"
        )

        # 8. Log round results
        round_time = time.time() - round_start
        round_data = {
            'round': round_num,
            'labeled_count': len(pool_manager.get_labeled()),
            'unlabeled_count': len(pool_manager.get_unlabeled()),
            'queries_this_round': len(escalated_ids),
            'total_queries': len(pool_manager.get_labeled()) - len(pool_manager.seed_df),
            'auto_accepted_this_round': len(auto_accept_idx),
            'unsafe_auto_accepts': unsafe_count,
            'round_time_seconds': round_time,
            'uncertainty_threshold_used': effective_unc_threshold,
            'risk_threshold_used': effective_risk_threshold,
            'query_budget_used': effective_query_budget,
            'use_dynamic_weights': use_dynamic_weights,
            **metrics,
        }
        round_results.append(round_data)

        # Learning curve for THIS experiment specifically — overwritten
        # each round so it's always current, available without --run-all.
        plot_experiment_learning_curve(
            round_results, experiment_id, save_dir=experiment_plots_dir
        )

        print(f"  Accuracy: {metrics['accuracy']:.4f} | "
              f"F1 Macro: {metrics['f1_macro']:.4f} | "
              f"FN Rate (malignant): {metrics['fn_rate_malignant']:.4f} | "
              f"Unsafe auto-accepts: {unsafe_count}")

        # 9. Save checkpoint (model + pool state + results metadata)
        round_ckpt_dir = os.path.join(
            _get_checkpoint_dir(experiment_id), f"round_{round_num}"
        )
        os.makedirs(round_ckpt_dir, exist_ok=True)

        # Save model weights (both heads + backbone)
        model.save_checkpoint(os.path.join(round_ckpt_dir, "model.pt"))

        # Save pool state (labeled/unlabeled/test CSVs)
        pool_manager.save_state(os.path.join(round_ckpt_dir, "pool_state"))

        # Save metadata for resume (completed round + accumulated results).
        # Thresholds are NOT reloaded on resume (they're recalibrated fresh
        # every round regardless), so they're recorded here only for
        # per-round reference/debugging, not read back by run_experiment().
        meta = {
            'experiment_id': experiment_id,
            'model_name': model_name,
            'uncertainty_method': uncertainty_method,
            'policy_name': policy_name,
            'completed_round': round_num,
            'num_rounds': num_rounds,
            'calibrated_unc_threshold_this_round': calibrated_unc_threshold,
            'calibrated_risk_threshold_this_round': calibrated_risk_threshold,
            'manual_risk_threshold': manual_risk_threshold,
            'query_budget': effective_query_budget,
            'use_dynamic_weights': use_dynamic_weights,
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
        os.makedirs(experiment_dir, exist_ok=True)
        results_df = pd.DataFrame(round_results)
        results_df.to_csv(
            os.path.join(experiment_dir, "results.csv"),
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
    with open(os.path.join(experiment_dir, "full.json"), 'w') as f:
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
