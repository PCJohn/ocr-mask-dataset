"""Minimal PyTorch Dataset example showing how to load the unified
images+masks format produced by build_dataset.py, upscaling the
low-res stored mask back to the image resolution.

Not required to run build_dataset.py; just a usage reference.
"""
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.common import read_jsonl


class OCRSegDataset(Dataset):
    def __init__(self, processed_dir: str, manifest_name: str = "manifest.jsonl", transform=None):
        self.processed_dir = processed_dir
        self.records = read_jsonl(os.path.join(processed_dir, manifest_name))
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        base = os.path.join(self.processed_dir, rec["dataset"])
        img = cv2.imread(os.path.join(base, rec["image_path"]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(base, rec["mask_path"]), cv2.IMREAD_GRAYSCALE)

        # upscale mask (nearest-neighbor, it's binary) to match the image size
        mask = cv2.resize(mask, (rec["resized_width"], rec["resized_height"]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        sample = {"image": img, "mask": mask, "id": rec["id"], "dataset": rec["dataset"]}
        if self.transform:
            sample = self.transform(sample)
        return sample
