import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        filename,
        data_dir: Path = Path("data/processed"),
        image_size: int = 128,
        augment: bool = False,
    ):
        df = pd.read_csv(data_dir / filename)
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _load(self, idx):
        return Image.open(self.paths[idx]).convert("RGB")

    def _augment(self, image):
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.uniform(-15, 15)
        image = image.rotate(angle, fillcolor=(0, 0, 0))
        return image

    def __getitem__(self, idx):
        image = self._load(idx)
        if self.augment:
            image = self._augment(image)
        size = self.image_size
        image = image.resize((size, size), Image.BILINEAR)
        x = np.array(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


class CNNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)