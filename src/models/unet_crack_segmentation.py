import os
import sys
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import argparse
from COSED import COSED
from smp_model import SMP_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--episodes", default=5, type=int, help="Training episodes.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--architecture", default='fpn', type=str, help="Which architecture to use")
parser.add_argument("--encoder", default='efficientnet-b6', type=str, help="Which encoder to use")
parser.add_argument('--print_images', default=False, type=bool)
parser.add_argument('--loss', default='bce', type=str, choices=['bce', 'dice'])
parser.add_argument('--t', default=0.5, type=float)

def augment_dataset(datum:dict [str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    transformation = v2.Compose([
        v2.Resize((416,416)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(40),
        #v2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)), #Dataset might contain some noise
        v2.ToDtype(torch.float32)
    ])

    image = datum['image']
    mask  = datum['mask']

    state = torch.get_rng_state()
    image = transformation(image)

    torch.set_rng_state(state)
    mask = transformation(mask)

    return image, mask


def eval_augment(datum:dict [str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    transformation = v2.Compose([
        v2.Resize((416,416)),
        #v2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32)
    ])

    image = datum['image']
    mask  = datum['mask']

    image = transformation(image)
    mask  = transformation(mask)

    return image, mask


def main(args: argparse.Namespace) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} to train the model')

    ####################
    # DATA PREPARATION #
    ####################

    csv_file_dir = '../../datasets/crackSeg9/image_mask_pairs.csv'
    root_dir     = '../../datasets/crackSeg9'

    dataset = COSED.Dataset(csv_file=csv_file_dir,
                            root_dir=root_dir,
                            #preload=True,
                            )
    train, dev, test = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

    train = COSED.TransformedDataset(train, transform=augment_dataset)
    dev   = COSED.TransformedDataset(dev, transform=eval_augment)
    test  = COSED.TransformedDataset(test, transform=eval_augment)

    # Load the data into Dataloaders
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader   = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #####################
    # VARS FOR TRAINING #
    #####################
    EPOCHS      = args.episodes
    T_MAX       = EPOCHS * len(train_loader)
    OUT_CLASSES = 1
    ARCH        = args.architecture
    ENCODER     = args.encoder
    THRESHOLD   = args.t

    ##########
    # LOGDIR #
    ##########

    model_name  = f'{ARCH}_{ENCODER}_bs{args.batch_size}_ep{args.episodes}_lr{args.learning_rate}_loss_{args.loss}'
    args.logdir = os.path.join('logs', '{}-{}-{}'.format(
        os.path.basename(globals().get('__file__', 'notebook')),
        model_name,
        datetime.datetime.now().strftime("%m-%d_%H%M")
    ))

    ###########################
    # NETWORK INIT + TRAINING #
    ###########################

    model = SMP_model(ARCH,
                      ENCODER,
                      in_channels=3,
                      out_classes=OUT_CLASSES,
                      T_MAX=T_MAX,
                      threshold=THRESHOLD,
                      device=device,
                      logdir=args.logdir,
                      dynamic_pos_weight=True,
                      loss=args.loss)

    print(f'Training model {model_name} \n------------------')
    model.train_model(train_loader, dev_loader, EPOCHS, args.learning_rate)
    model.test_model(test_loader)

    ####################
    # SAVING THE MODEL #
    ####################
    saving_to = f'./{args.logdir}/{ARCH}_{ENCODER}'
    model.model.save_pretrained(saving_to)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
