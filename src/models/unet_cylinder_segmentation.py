import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, v2
import torchvision
import torchmetrics
import argparse
import segmentation_models_pytorch as smp
from COSED import COSED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from smp_model import SMP_model

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--episodes", default=10, type=int, help="Training episodes.")
parser.add_argument("--res", default=128, type=int, help="square resolution of the image")
parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate.")

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
            v2.RandomChannelPermutation()
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
        ])

    img  = datum['image']
    mask = datum['mask']

    return transformation(img), transformation(mask)


def main(args: argparse.Namespace) -> None:
    dataset = COSED.Dataset(csv_file='./dataset/concrete_segmentation_camvid/test2.txt',
                            root_dir='./dataset/concrete_segmentation_camvid')

    train, dev, test = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

    train_set = COSED.TransformedDataset(train, transform=augment_dataset)
    dev_set   = COSED.TransformedDataset(dev, transform=eval_augment)
    test_set  = COSED.TransformedDataset(test, transform=eval_augment)


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader   = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    EPOCHS      = args.episodes
    T_MAX       = EPOCHS * len(train_loader)
    OUT_CLASSES = 1
    ARCH        = 'unet'
    ENCODER     = 'mit_b0'

    # model = CamVidModel('unet', 'efficientnet-b0', in_channels=3, out_classes=OUT_CLASSES, T_MAX=T_MAX)
#
    # trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)
#
    # trainer.fit(
    #     model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=dev_loader,
    # )


    model = SMP_model(ARCH, ENCODER, in_channels=3, out_classes=OUT_CLASSES, T_MAX=T_MAX, threshold=0.5)
    model.train_model(train_loader, dev_loader, EPOCHS, args.learning_rate)
    model.test_model(test_loader)

    ### Print some images
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    i = 0
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
        pr_masks = (pr_masks > 0.2).type(torch.uint8) #apply thresholding

        for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
            if idx <= 5:
                imshow(title=f'{i}_{idx}',
                       image=image.cpu(),
                       ground_truth=gt_mask.squeeze().cpu(),
                       prediction=pr_mask.squeeze().cpu())
            else:
                break

    model_name = f'{ARCH}_{ENCODER}_bs{args.batch_size}_ep{args.episodes}_res{args.res}'
    saving_to = f'./weights/{model_name}'
    torch.save(model.state_dict(), saving_to)

#    #resnext50_32x4d
    #model = CamVidModel('unet', 'efficientnet-b0', in_channels=3, out_classes=OUT_CLASSES, T_MAX=T_MAX)
#
    #trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)
#
    #trainer.fit(
    #    model,
    #    train_dataloaders=train_loader,
    #    val_dataloaders=dev_loader,
    #)
#
#    valid_metrics = trainer.validate(model, dataloaders=dev_loader, verbose=False)
#    print(valid_metrics)
#
#    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
#    print(test_metrics)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
