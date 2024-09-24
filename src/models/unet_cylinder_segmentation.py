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

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--episodes", default=10, type=int, help="Training episodes.")
parser.add_argument("--res", default=128, type=int, help="square resolution of the image")
#parser.add_argument("--hidden_layer_size", default=..., type=int, help="Size of hidden layer.")
#parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
#parser.add_argument("--dropout", default=..., type=float, help="Dropout")

def imshow(title= None, **images):
    """Displays images in one row"""
    n = len(images)
    #plt.figure(figsize=(5,5))
    plt.figure(figsize=(n*5,5))

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
    plt.pause(0.001)

def augment_dataset(datum: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    transformation = v2.Compose([
            v2.Resize((args.res,args.res)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(80),
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
    img = datum['image']
    mask = datum['mask']

    return transformation(img), transformation(mask)

class CamVidModel(pl.LightningModule):
    def __init__(self, architecture, encoder_name, in_channels, out_classes, T_MAX, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params['std']).view(1,3,1,1))
        self.register_buffer("mean", torch.tensor(params['mean']).view(1,3,1,1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.T_MAX = T_MAX

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Image shape should be (bs, num_channels, h, w)
        assert image.ndim == 4

        # Check that image dims are divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Checking that the mask is valid
        assert mask.ndim == 4
        assert mask.max() <= 1 and mask.min() >= 0

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)

        # Metrics with thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).type(torch.uint8)

        # Calculating tp, fp, fn, tn
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode='binary'
        )
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro-imagewise')
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')

        self.log('l', loss, prog_bar=True, on_step=True, logger=True)
        #self.log('pr', precision, prog_bar=True, on_step=True, logger=True)
        self.log('f1', f1, prog_bar=True, on_step=True, logger=True)

        return{
            "loss": loss,
            "tp":tp,
            "fp":fp,
            "fn":fn,
            "tn":tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # First calculate IoU for each image and then compute mean
        # over those scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')

        metrics     = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f'{stage}_dataset_f1':dataset_f1
        }
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, 'train')
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, 'valid')
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, 'valid')
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, 'test')
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_MAX, eta_min=1e-5)
        return {
            'optimizer':optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

def main(args: argparse.Namespace) -> None:
    dataset = COSED.Dataset(csv_file='./dataset/concrete_segmentation_camvid/test2.txt',
                            root_dir='./dataset/concrete_segmentation_camvid')

    train, dev, test = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

    train_set = COSED.TransformedDataset(train, transform=augment_dataset)
    dev_set   = COSED.TransformedDataset(dev, transform=eval_augment)
    test_set  = COSED.TransformedDataset(test, transform=eval_augment)

    img, mask = train_set[0]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    EPOCHS      = args.episodes
    T_MAX       = EPOCHS * len(train_loader)
    OUT_CLASSES = 1

    #resnext50_32x4d
    model = CamVidModel('unet', 'efficientnet-b0', in_channels=3, out_classes=OUT_CLASSES, T_MAX=T_MAX)

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=dev_loader,
    )

    valid_metrics = trainer.validate(model, dataloaders=dev_loader, verbose=False)
    print(valid_metrics)

    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)

    images, masks = next(iter(test_loader))

    model.eval()
    with torch.no_grad():
        logits = model(images)

    pr_masks = logits.sigmoid()
    #pr_masks = (pr_masks > 0.3)

    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx <= 5:
            imshow(title=str(idx),
                   image=image,
                   ground_truth=gt_mask.squeeze(),
                   prediction=pr_mask.squeeze())

        else:
            break


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
