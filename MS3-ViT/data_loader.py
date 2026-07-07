"""
HSI Data Loader for MS3-ViT
============================
On-the-fly patch extraction (memory-efficient).
Loads .mat HSI datasets, splits known/unknown classes,
extracts spatial patches on-demand, and applies data augmentation.

Supports: Indian Pines, Pavia University, Salinas
"""

import os
import random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Data Augmentation
# ============================================================

class HSIAugmentation:
    """HSI-specific data augmentation (operates on numpy arrays)."""

    @staticmethod
    def random_flip(cube):
        if random.random() > 0.5:
            cube = np.flip(cube, axis=-2).copy()
        if random.random() > 0.5:
            cube = np.flip(cube, axis=-1).copy()
        return cube

    @staticmethod
    def random_rotate90(cube):
        k = random.randint(0, 3)
        if k > 0:
            cube = np.rot90(cube, k=k, axes=(-2, -1)).copy()
        return cube

    @staticmethod
    def spectral_noise(cube, std=0.03):
        noise = np.random.randn(*cube.shape).astype(np.float32) * std
        return (cube + noise).astype(np.float32)

    @staticmethod
    def spatial_dropout(cube, drop_ratio=0.15):
        mask = (np.random.rand(*cube.shape[-2:]) > drop_ratio).astype(np.float32)
        return cube * mask[np.newaxis, :, :]

    @classmethod
    def weak_augment(cls, cube):
        cube = cls.random_flip(cube)
        cube = cls.random_rotate90(cube)
        return cube

    @classmethod
    def strong_augment(cls, cube):
        cube = cls.random_flip(cube)
        cube = cls.random_rotate90(cube)
        cube = cls.spectral_noise(cube, std=0.03)
        cube = cls.spatial_dropout(cube, drop_ratio=0.15)
        return cube


# ============================================================
# Memory-Efficient HSI Dataset (patches extracted on-the-fly)
# ============================================================

class HSIDatasetOnTheFly(Dataset):
    """
    PyTorch Dataset that extracts HSI patches on-the-fly.

    Stores only pixel coordinates — patches extracted in __getitem__.
    Memory usage: O(S*S*C + N_coords) instead of O(N_patches * C * S * S).
    """

    def __init__(self, data, coords, labels=None, patch_size=13,
                 is_labeled=True, use_augmentation=True, pad_mode='edge'):
        """
        Args:
            data: (H, W, C) float32 HSI data array
            coords: list of (i, j) pixel coordinates
            labels: (N,) array of class labels, or None for unlabeled
            patch_size: spatial patch size S
            is_labeled: whether these are labeled samples
            use_augmentation: apply data augmentation
        """
        self.data = np.ascontiguousarray(data)
        self.H, self.W, self.C = data.shape
        self.coords = coords
        self.labels = labels
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.is_labeled = is_labeled
        self.use_augmentation = use_augmentation
        self.aug = HSIAugmentation()
        self.pad_mode = pad_mode

        # Pre-pad the full data for fast patch extraction
        # Using 'edge' mode (memory-efficient) instead of 'reflect'
        self.data_padded = np.pad(
            self.data,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode=self.pad_mode,
        ).copy()  # make contiguous

        # Pre-allocate patch buffer per worker (reused each __getitem__)
        self._patch_buffer = np.zeros((self.C, patch_size, patch_size),
                                       dtype=np.float32)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        pi, pj = i + self.pad, j + self.pad

        # Extract patch from padded data: (S, S, C) -> (C, S, S)
        patch = self.data_padded[pi - self.pad:pi + self.pad + 1,
                                  pj - self.pad:pj + self.pad + 1, :]
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1))

        if self.is_labeled:
            if self.use_augmentation:
                patch = self.aug.weak_augment(patch)
            label = self.labels[idx]
            return torch.from_numpy(patch).float(), torch.tensor(label).long()
        else:
            if self.use_augmentation:
                weak = self.aug.weak_augment(patch.copy())
                strong = self.aug.strong_augment(patch)
                return (torch.from_numpy(weak).float(),
                        torch.from_numpy(strong).float())
            else:
                t = torch.from_numpy(patch).float()
                return t, t


# ============================================================
# HSI Data Manager
# ============================================================

class HSIDataManager:
    """Manages HSI data: loading, preprocessing, train/test/unlabeled split."""

    def __init__(self, dataset_name, data_dir, config):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.config = config

        from config import DATASET_CONFIGS
        self.ds_config = DATASET_CONFIGS[dataset_name]

        self.num_labeled = config.get("num_labeled_per_class", 10)
        self.batch_size_labeled = config.get("batch_size_labeled", 32)
        self.batch_size_unlabeled = config.get("batch_size_unlabeled", 32)
        self.patch_size = self.ds_config.get("spatial_size", 13)

        self._load_data()
        self._preprocess()

    def _load_data(self):
        mat_path = os.path.join(self.data_dir, self.ds_config["mat_file"])
        gt_path = os.path.join(self.data_dir, self.ds_config["gt_file"])
        mat = sio.loadmat(mat_path)
        gt = sio.loadmat(gt_path)
        self.data = mat[self.ds_config["mat_key"]].astype(np.float32)
        self.gt = gt[self.ds_config["gt_key"]].astype(np.int64)
        self.height, self.width, self.num_bands = self.data.shape
        self.num_classes = self.ds_config["num_classes"]

    def _preprocess(self):
        for i in range(self.num_bands):
            band = self.data[:, :, i]
            b_min, b_max = band.min(), band.max()
            if b_max > b_min:
                self.data[:, :, i] = (band - b_min) / (b_max - b_min)

    def set_open_set_split(self, unknown_classes):
        self.unknown_classes = unknown_classes
        self.known_classes = [c for c in range(1, self.num_classes + 1)
                              if c not in unknown_classes]

    def generate_splits(self, unknown_classes, num_labeled=None, seed=42):
        if num_labeled is None:
            num_labeled = self.num_labeled
        rng = np.random.RandomState(seed)
        self.set_open_set_split(unknown_classes)

        train_indices, test_indices, unlabeled_indices = [], [], []

        for c in range(1, self.num_classes + 1):
            coords = np.argwhere(self.gt == c)
            N_c = len(coords)

            if c in unknown_classes:
                test_indices.extend([(i, j) for i, j in coords])
            else:
                perm = rng.permutation(N_c)
                train_indices.extend([(i, j) for i, j in coords[perm[:num_labeled]]])
                test_count = min(200, max(num_labeled, N_c - num_labeled))
                test_indices.extend([(i, j) for i, j in coords[perm[num_labeled:num_labeled + test_count]]])
                unlabeled_indices.extend([(i, j) for i, j in coords[perm[num_labeled + test_count:]]])

        # Background pixels -> unlabeled pool (sampled)
        bg_coords = np.argwhere(self.gt == 0)
        max_bg = min(len(bg_coords), 30000)
        if len(bg_coords) > max_bg:
            bg_coords = bg_coords[rng.permutation(len(bg_coords))[:max_bg]]
        unlabeled_indices.extend([(i, j) for i, j in bg_coords])

        self.splits = {
            "train": train_indices,
            "test": test_indices,
            "unlabeled": unlabeled_indices,
        }
        return self.splits

    def get_dataloaders(self, splits=None):
        if splits is None:
            splits = self.splits

        # Build label remapping
        label_remap = {}
        for new_idx, orig in enumerate(self.known_classes):
            label_remap[orig] = new_idx
        for uc in self.unknown_classes:
            label_remap[uc] = self.num_classes

        # Training labels
        train_labels = np.array([label_remap.get(self.gt[i, j], -1)
                                  for i, j in splits["train"]], dtype=np.int64)
        valid = train_labels >= 0
        train_coords = [(i, j) for k, (i, j) in enumerate(splits["train"]) if valid[k]]
        train_labels = train_labels[valid]

        # Test labels
        test_labels = np.array([label_remap.get(self.gt[i, j], 0)
                                 for i, j in splits["test"]], dtype=np.int64)

        print(f"  Train: {len(train_coords)} labeled samples")
        print(f"  Test: {len(splits['test'])} samples")
        print(f"  Unlabeled: {len(splits['unlabeled'])} samples")

        # Create on-the-fly datasets
        train_dataset = HSIDatasetOnTheFly(
            self.data, train_coords, train_labels,
            patch_size=self.patch_size, is_labeled=True,
        )
        test_dataset = HSIDatasetOnTheFly(
            self.data, splits["test"], test_labels,
            patch_size=self.patch_size, is_labeled=True,
            use_augmentation=False,
        )
        unlabeled_dataset = HSIDatasetOnTheFly(
            self.data, splits["unlabeled"], None,
            patch_size=self.patch_size, is_labeled=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size_labeled,
                                  shuffle=True, drop_last=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size_labeled * 2,
                                 shuffle=False, num_workers=0)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.batch_size_unlabeled,
                                      shuffle=True, drop_last=True, num_workers=0)

        return train_loader, test_loader, unlabeled_loader
