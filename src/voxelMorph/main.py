import torch
import torchvision
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from COCIRD import COCIRD
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--res', default=512, type=int)

def augment_dataset(datum: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    trans = v2.Compose([
            v2.Resize((args.res, args.res)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToDtype(torch.float32),
        ])

    fixed  = datum['fixed']
    moving = datum['moving']

    state = torch.get_rng_state()
    fixed = trans(fixed)

    torch.set_rng_state(state)
    moving = trans(moving)

    return moving, fixed


def eval_augment(datum: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    trans = v2.Compose([
        v2.Resize((args.res, args.res)),
        v2.ToDtype(torch.float32)
    ])

    moving = datum['moving']
    fixed  = datum['fixed']

    return trans(moving), trans(fixed)


def main(args):
    #Using GPU if able
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################
    # DATA PREPARATION #
    ####################
    data_csv = '../../dev_dataset/pairwise_permutations.csv'
    data_root = '../../dev_dataset'
    dataset = COCIRD.Dataset(csv_file=data_csv,
                             root_dir=data_root)

    train, dev = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_set = COCIRD.TransformedDataset(train, transform=augment_dataset)
    dev_set   = COCIRD.TransformedDataset(dev, transform=eval_augment)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    dev_loader   = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=6)

    ##########################
    # VARIABLES FOR TRAINING #
    ##########################
    EPOCHS = 10

    # INIT MODEL #

    ############
    # TRAINING #
    ############
    for epoch in range(EPOCHS):
        train_loss = 0
        val_loss = 0
        for batch_moving, batch_fixed in train_loader:
            loss = model.train_step(batch_moving, batch_fixed)
            train_loss += loss.data
        print(f'train loss after epoch: {train_loss * args.batch_size / len(train_loader)}')

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

