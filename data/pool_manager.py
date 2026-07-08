"""
Pool Manager: manages the labeled / unlabeled / test split for active learning.

Data flow:
  Seed (490) → initial labeled set
  Remaining (9,525) → 80% unlabeled pool (~7,620) + 20% test set (~1,905)

During AL rounds, images move from unlabeled pool → labeled set.
The test set is NEVER touched — used only for evaluation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import RANDOM_SEED, TEST_SPLIT_RATIO


class PoolManager:
    """
    Manages the three data pools for active learning:
    - labeled_df: images with known labels (starts with seed data)
    - unlabeled_df: images available for oracle querying
    - test_df: held-out test set (never modified)

    Parameters
    ----------
    seed_csv : str
        Path to seed_metadata.csv (490 labeled images).
    remaining_csv : str
        Path to remaining metadata CSV (9,525 images).
    test_ratio : float
        Fraction of remaining data to hold out as test set.
    random_seed : int
        Random seed for reproducible train/test split.
    """

    def __init__(self, seed_csv, remaining_csv,
                 test_ratio=TEST_SPLIT_RATIO, random_seed=RANDOM_SEED):

        self.seed_df = pd.read_csv(seed_csv)
        remaining_df = pd.read_csv(remaining_csv)

        # Split remaining into unlabeled pool (80%) and test set (20%)
        # Stratified split to maintain class proportions in both
        self.unlabeled_df, self.test_df = train_test_split(
            remaining_df,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=remaining_df['dx']
        )
        self.unlabeled_df = self.unlabeled_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        # Labeled set starts with seed data
        self.labeled_df = self.seed_df.copy()

        # Track query history
        self.query_history = []  # list of dicts per round

        print(f"Pool Manager initialized:")
        print(f"  Labeled (seed):   {len(self.labeled_df)} images")
        print(f"  Unlabeled pool:   {len(self.unlabeled_df)} images")
        print(f"  Test set:         {len(self.test_df)} images")

    def get_labeled(self):
        """Return the current labeled DataFrame."""
        return self.labeled_df.copy()

    def get_unlabeled(self):
        """Return the current unlabeled pool DataFrame."""
        return self.unlabeled_df.copy()

    def get_test(self):
        """Return the fixed test set DataFrame."""
        return self.test_df.copy()

    def move_to_labeled(self, image_ids, labels_dict=None):
        """
        Move images from unlabeled pool to labeled set.

        Parameters
        ----------
        image_ids : list of str
            Image IDs to move (e.g., ['ISIC_0027419', 'ISIC_0025030']).
        labels_dict : dict, optional
            Mapping of image_id → dx. If None, labels are taken from
            the unlabeled_df (they already have ground-truth labels).
        """
        # Find rows in unlabeled pool
        mask = self.unlabeled_df['image_id'].isin(image_ids)
        moving = self.unlabeled_df[mask].copy()

        if len(moving) == 0:
            print("WARNING: No matching images found in unlabeled pool.")
            return

        # Add to labeled set
        self.labeled_df = pd.concat(
            [self.labeled_df, moving], ignore_index=True
        )

        # Remove from unlabeled pool
        self.unlabeled_df = self.unlabeled_df[~mask].reset_index(drop=True)

        # Record query
        self.query_history.append({
            'round': len(self.query_history) + 1,
            'queries_this_round': len(moving),
            'total_labeled': len(self.labeled_df),
            'remaining_unlabeled': len(self.unlabeled_df),
        })

    def get_stats(self):
        """Return current pool statistics."""
        stats = {
            'labeled_count': len(self.labeled_df),
            'unlabeled_count': len(self.unlabeled_df),
            'test_count': len(self.test_df),
            'labeled_distribution': self.labeled_df['dx'].value_counts().to_dict(),
            'unlabeled_distribution': self.unlabeled_df['dx'].value_counts().to_dict(),
            'total_queries': len(self.labeled_df) - len(self.seed_df),
        }
        return stats

    def get_query_history(self):
        """Return the full query history as a DataFrame."""
        if not self.query_history:
            return pd.DataFrame()
        return pd.DataFrame(self.query_history)

    def save_state(self, save_dir):
        """Save current state to disk for checkpoint resumption."""
        os.makedirs(save_dir, exist_ok=True)
        self.labeled_df.to_csv(
            os.path.join(save_dir, "labeled.csv"), index=False
        )
        self.unlabeled_df.to_csv(
            os.path.join(save_dir, "unlabeled.csv"), index=False
        )
        self.test_df.to_csv(
            os.path.join(save_dir, "test.csv"), index=False
        )
        if self.query_history:
            pd.DataFrame(self.query_history).to_csv(
                os.path.join(save_dir, "query_history.csv"), index=False
            )
        print(f"Pool state saved to {save_dir}")

    def load_state(self, save_dir):
        """Load state from a checkpoint directory."""
        self.labeled_df = pd.read_csv(
            os.path.join(save_dir, "labeled.csv")
        )
        self.unlabeled_df = pd.read_csv(
            os.path.join(save_dir, "unlabeled.csv")
        )
        self.test_df = pd.read_csv(
            os.path.join(save_dir, "test.csv")
        )
        history_path = os.path.join(save_dir, "query_history.csv")
        if os.path.exists(history_path):
            self.query_history = pd.read_csv(history_path).to_dict('records')
        print(f"Pool state loaded from {save_dir}")
        print(f"  Labeled: {len(self.labeled_df)} | "
              f"Unlabeled: {len(self.unlabeled_df)} | "
              f"Test: {len(self.test_df)}")
