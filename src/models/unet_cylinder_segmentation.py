import os
import re
import datetime
import pytz
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import argparse
from COSED import COSED
import matplotlib.pyplot as plt
from smp_model import SMP_model
import segmentation_models_pytorch as smp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--episodes", default=3, type=int, help="Training episodes.")
parser.add_argument("--res", default=1024, type=int, help="square resolution of the image")
parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate.")
parser.add_argument("--architecture", default='unet', type=str, help="Which architecture to use")
parser.add_argument("--encoder", default='mit_b0', type=str, help="Which encoder to use")
parser.add_argument('--print_images', default=False, type=bool)

def imshow(title= None, **images):
    """Displays images in one row"""
    n = len(images)
    plt.figure(figsize=(n*4,5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name=='image':
            plt.imshow(image.permute(1, 2, 0))
        else:
            plt.imshow(image)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('result' if title is None else title)

    plt.show()
    plt.pause(0.1)

def augment_dataset(datum: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    transformation = v2.Compose([
            v2.Resize((args.res,args.res)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=30),
            v2.RandomChannelPermutation(),
            v2.ToDtype(torch.float32),
        ])

    image = datum['image']
    mask  = datum['mask']

    state = torch.get_rng_state()
    image = transformation(image)

    torch.set_rng_state(state)
    mask = transformation(mask)

    return image, mask

def eval_augment(datum: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    transformation = v2.Compose([
            v2.Resize((args.res,args.res)),
            v2.ToDtype(torch.float32),
        ])

    img  = datum['image']
    mask = datum['mask']

    return transformation(img), transformation(mask)


def main(args: argparse.Namespace) -> None:
    # Setting up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################
    # DATA PREPARATION #
    ####################
    csv_file_dir = '../../datasets/concrete_segmentation_camvid/image_mask_pairs.txt'
    root_dir = '../../datasets/concrete_segmentation_camvid'
    dataset = COSED.Dataset(csv_file=csv_file_dir,
                            root_dir=root_dir)
    #dataset = COSED.LMDB_Dataset(lmdb_path='test_lmdb')

    train, dev, test = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

    train_set = COSED.TransformedDataset(train, transform=augment_dataset)
    dev_set   = COSED.TransformedDataset(dev, transform=eval_augment)
    test_set  = COSED.TransformedDataset(test, transform=eval_augment)

    # Loading the data into DataLoaders, num_workers can be tweaked per device
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    dev_loader   = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=6)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=6)

    ##########################
    # VARIABLES FOR TRAINING #
    ##########################
    EPOCHS      = args.episodes
    T_MAX       = EPOCHS * len(train_loader)
    OUT_CLASSES = 1
    ARCH        = args.architecture
    ENCODER     = args.encoder
    THRESHOLD   = 0.5

    ##########
    # LOGDIR #
    ##########
    model_name = f'{ARCH}_{ENCODER}_bs{args.batch_size}_ep{args.episodes}_r{args.res}_lr{args.learning_rate}'
    # Taken from NPFL138 practicals
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        model_name,
        datetime.datetime.now(pytz.timezone('CET')).strftime("%m-%d_%H%M")
    ))

    ####################
    # NETWORK TRAINING #
    ####################
    model = SMP_model(ARCH,
                      ENCODER,
                      in_channels=3,
                      out_classes=OUT_CLASSES,
                      T_MAX=T_MAX,
                      threshold=THRESHOLD,
                      device=device,
                      logdir=args.logdir)

    print(f"Training model {model_name}\n-------------------")
    model.train_model(train_loader, dev_loader, EPOCHS, args.learning_rate)
    model.test_model(test_loader)

    ############
    # SHOWCASE #
    ############
    if args.print_images:
        model.eval()
        i = 0
        print("Printing images:", end='')
        for batch in test_loader:
            if i >= 5:
                break
            i += 1
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                logits = model(images)

            pr_masks = logits.sigmoid()
            #pr_masks = (pr_masks > 0.5).type(torch.uint8) #apply thresholding
            pr_masks = torch.clip(pr_masks, 0, 1)

            for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
                if idx <= 5:
                    imshow(title=f'{i}_{idx}',
                           image=image.cpu(),
                           ground_truth=gt_mask.squeeze().cpu(),
                           prediction=pr_mask.squeeze().cpu())
                else:
                    break
        print("")

    ####################
    # SAVING THE MODEL #
    ####################
    saving_to = f'./{args.logdir}/{ARCH}_{ENCODER}'
    model.model.save_pretrained(saving_to)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
