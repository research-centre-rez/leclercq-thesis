import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision
from typing import Any, Callable

import warnings

warnings.filterwarnings("ignore")

class COSED():
    class Dataset(Dataset):
        #COncrete SEgmentation Dataset
        def __init__(self, csv_file, root_dir) -> None:
            #self.transform = transform
            self.data = pd.read_csv(csv_file, header=None, names=['img', 'mask'])
            print("Example of data:")
            print(self.data.head())
            print("----------------")
            self.root_dir = root_dir
            self._size = len(self.data)

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index):
            if torch.is_tensor(index):
                index = index.item()

            img_name  = os.path.join(self.root_dir, self.data.iloc[index, 0])
            mask_name = os.path.join(self.root_dir, self.data.iloc[index, 1])

            # print(img_name)
            # print("----------------")
            # print(mask_name)
            # print("----------------")

            image = torchvision.transforms.PILToTensor()(Image.open(img_name).convert('RGB'))
            mask  = torchvision.transforms.PILToTensor()(Image.open(mask_name).convert('L'))

            sample = {'image': image, 'mask': mask}

            return sample

        def transform(self, transform: Callable[[dict[str, torch.Tensor]], Any]) -> Dataset:
            return COSED.TransformedDataset(self, transform)

    class TransformedDataset(Dataset):
        def __init__(self, dataset: "COSED.Dataset", transform: Callable[[dict[str, torch.Tensor]], Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)
            #return self._dataset._size

        def __getitem__(self, index: int) -> Any:
            return self._transform(self._dataset[index])

        def transform(self, transform: Callable[[dict[str, torch.Tensor]], Any]) -> torch.utils.data.Dataset:
            return COSED.TransformedDataset(self, transform)
