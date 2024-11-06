import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision
from typing import Any, Callable
import lmdb
import cv2 as cv
import numpy as np
from tqdm import tqdm 
import sys

import warnings

from torchvision.transforms import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

warnings.filterwarnings("ignore")

class COSED():
    """
    This the COncrete SEgmentation Dataset. This version takes in a csv files that contains
    image-mask pairs that are then read into the memory on the fly.
    Args:
        csv_file: relative path to the csv file
        root_dir: relative path to the root of the dataset
        threshold: in the case of noisy masks a basic threshold is applied to denoise them
    """
    class Dataset(Dataset):
        def __init__(self, csv_file, root_dir, threshold=230, preload=False) -> None:
            print(f'You are using the base version of COSED {"with" if preload else "without"} preloading')
            self.data     = pd.read_csv(csv_file, header=None, names=['img', 'mask'])
            self.root_dir = root_dir
            self._size    = len(self.data)
            self.t        = threshold #in case there are noisy images we do basic thresholding
            self.cached   = preload

            if self.cached:
                print('Caching the dataset into memory..')
                self.images = []
                self.masks  = []

                for _, row in tqdm(self.data.iterrows(), total=self._size, desc='Loading'):
                    img_name  = os.path.join(self.root_dir, row['img'])
                    mask_name = os.path.join(self.root_dir, row['mask'])

                    image = Image.open(img_name).convert('RGB')
                    image = torchvision.transforms.PILToTensor()(image)

                    mask  = torchvision.transforms.PILToTensor()(Image.open(mask_name).convert('L'))
                    mask  = (mask > self.t).type(torch.uint8)

                    self.images.append(image)
                    self.masks.append(mask)

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index):
            if torch.is_tensor(index):
                index = index.item()

            if self.cached:
                image = self.images[index]
                mask  = self.masks[index]
            else:
                img_name  = os.path.join(self.root_dir, self.data.iloc[index, 0])
                mask_name = os.path.join(self.root_dir, self.data.iloc[index, 1])

                image = Image.open(img_name).convert('RGB')
                image = torchvision.transforms.PILToTensor()(image)

                pil_mask = Image.open(mask_name).convert('L')
                tensor_mask = torchvision.transforms.PILToTensor()(pil_mask)
                mask  = torchvision.transforms.PILToTensor()(Image.open(mask_name).convert('L'))
                mask  = (mask > self.t).type(torch.uint8)
                visualisers.imshow('mask_comparison', tensor_mask=tensor_mask.permute(1,2,0), pil_mask=pil_mask, mask=mask.permute(1,2,0))


            sample = {'image': image, 'mask': mask}

            return sample

        def transform(self, transform: Callable[[dict[str, torch.Tensor]], Any]) -> Dataset:
            return COSED.TransformedDataset(self, transform)


    class LMDB_Dataset(Dataset):
        def __init__(self, lmdb_path) -> None:
            print('You are using the LMDB version of COSED')
            self.env = lmdb.open(lmdb_path, readonly=True, lock=False)

            with self.env.begin() as txn:
                length = txn.get(b'__len__')
                if length is None:
                    raise ValueError("Dataset length key no found!!")
                self._size = int(length.decode())

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index):
            if torch.is_tensor(index):
                index = index.item()

            with self.env.begin(buffers=True) as txn:
                cursor = txn.cursor()
                img_key  = f'image_{index}'.encode()
                mask_key = f'mask_{index}'.encode()

                img_bytes  = cursor.get(img_key)
                mask_bytes = cursor.get(mask_key)

            img  = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
            mask = cv.imdecode(np.frombuffer(mask_bytes, np.uint8), cv.IMREAD_GRAYSCALE)

            image = transforms.ToTensor()(img)

            mask = transforms.ToTensor()(mask)
            mask = mask > 0

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

        def __getitem__(self, index: int) -> Any:
            return self._transform(self._dataset[index])

        def transform(self, transform: Callable[[dict[str, torch.Tensor]], Any]) -> torch.utils.data.Dataset:
            return COSED.TransformedDataset(self, transform)
