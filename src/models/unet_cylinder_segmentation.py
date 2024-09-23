import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, v2
import torchmetrics
import argparse
import segmentation_models_pytorch as smp
from COSED import COSED
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--episodes", default=10, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=..., type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--dropout", default=..., type=float, help="Dropout")

def augment_dataset(datum: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    transformation = v2.Compose([
            v2.Resize((512,512)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ])

    image = datum['image']
    mask  = datum['mask']

    state = torch.get_rng_state()
    # image = torch.permute(datum["image"], (2,0,1))
    image = transformation(image)
    # image = torch.permute(image, (1, 2, 0))

    torch.set_rng_state(state)
    # mask = torch.permute(datum["mask"], (2,0,1))
    mask = transformation(mask)
    # mask = torch.permute(mask, (1, 2, 0))
    return image, mask

def eval_augment(datum: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    transformation = v2.Compose([
            v2.Resize((512,512)),
        ])
    img = datum['image']
    mask = datum['mask']

    return transformation(img), transformation(mask)

def main(args: argparse.Namespace) -> None:
    u_net = smp.Unet()

    dataset = COSED.Dataset(csv_file='./dataset/concrete_segmentation_camvid/test2.txt',
                            root_dir='./dataset/concrete_segmentation_camvid')

    train, dev, test = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

    train_set = COSED.TransformedDataset(train, transform=augment_dataset)
    dev_set   = COSED.TransformedDataset(dev, transform=eval_augment)
    test_set  = COSED.TransformedDataset(test, transform=eval_augment)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    ...

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
