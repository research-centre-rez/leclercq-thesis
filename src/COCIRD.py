import os
import torch
from torch._dynamo.utils import istensor
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision
from typing import Any, Callable
import cv2 as cv
import numpy as np

import warnings
from torchvision.transforms import transforms

warnings.filterwarnings('ignore')

class COCIRD():
    """
    This is the COncrete Cylinders Image Registration Dataset
    """
    class Dataset(Dataset):
        def __init__(self, csv_file, root_dir) -> None:
            self.data = pd.read_csv(csv_file, header=None, names=['Fixed', 'Moving'])
            self.root_dir = root_dir
            self._size = len(self.data)

        def __len__(self) -> int:
            return self._size
        
        def __getitem__(self, index):
            if torch.is_tensor(index):
                index = index.item()

            fixed_name  = os.path.join(self.root_dir, self.data.iloc[index, 0])
            moving_name = os.path.join(self.root_dir, self.data.iloc[index, 1])

            fixed  = Image.open(fixed_name).convert('RGB')
            fixed  = torchvision.transforms.PILToTensor()(fixed)
            moving = Image.open(moving_name).convert('RGB')
            moving = torchvision.transforms.PILToTensor()(moving)

            sample = {'fixed': fixed, 'moving': moving}
            return sample

        def transform(self, transform: Callable[[dict[str, torch.Tensor]], Any]) -> Dataset:
            return COCIRD.TransformedDataset(self, transform)

    class TransformedDataset(Dataset):
        def __init__(self, dataset: "COCIRD.Dataset", transform: Callable[[dict[str, torch.Tensor]], Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            return self._transform(self._dataset[index])

        def transform(self, transform: Callable[[dict[str, torch.Tensor]], Any]) -> torch.utils.data.Dataset:
            return COCIRD.TransformedDataset(self, transform)

