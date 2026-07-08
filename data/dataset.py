"""
PyTorch Dataset class for HAM10000 skin lesion images.
Loads images from a directory based on a metadata CSV.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import CLASS_TO_IDX


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 dermoscopy images.

    Parameters
    ----------
    metadata_csv : str or pd.DataFrame
        Path to CSV file with columns: image_id, dx (diagnosis).
        Or a DataFrame directly.
    image_dirs : list of str
        List of directories to search for images.
    transform : callable, optional
        Torchvision transform to apply to each image.
    """

    def __init__(self, metadata_csv, image_dirs, transform=None):
        if isinstance(metadata_csv, pd.DataFrame):
            self.metadata = metadata_csv.reset_index(drop=True)
        else:
            self.metadata = pd.read_csv(metadata_csv)

        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.transform = transform

        # Build image path lookup for fast access
        self._image_paths = {}
        for _, row in self.metadata.iterrows():
            image_id = row['image_id']
            filename = f"{image_id}.jpg"
            for img_dir in self.image_dirs:
                full_path = os.path.join(img_dir, filename)
                if os.path.exists(full_path):
                    self._image_paths[image_id] = full_path
                    break

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        diagnosis = row['dx']
        label = CLASS_TO_IDX[diagnosis]

        # Load image
        image_path = self._image_paths.get(image_id)
        if image_path is None:
            raise FileNotFoundError(
                f"Image '{image_id}.jpg' not found in any of: {self.image_dirs}"
            )

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, image_id

    def get_labels(self):
        """Return all labels as a list of integers."""
        return [CLASS_TO_IDX[dx] for dx in self.metadata['dx']]

    def get_image_ids(self):
        """Return all image IDs as a list."""
        return self.metadata['image_id'].tolist()

    def get_class_distribution(self):
        """Return a dict of class name → count."""
        return self.metadata['dx'].value_counts().to_dict()
