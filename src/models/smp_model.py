import torch
import segmentation_models_pytorch as smp
import torchmetrics
from torch.optim import lr_scheduler
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class SMP_model(nn.Module):
    def __init__(self,
                 arch: str,
                 encoder_name: str,
                 in_channels: int,
                 out_classes: int,
                 T_MAX: int,
                 threshold,
                 device,
                 logdir,
                 dynamic_pos_weight = False,
                 loss = 'bce',
                 **kwargs):

        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        self.binary_threshold = threshold
        self.device = device

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        if loss == 'bce':
            pos_weight = torch.tensor([291/200])
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.loss = loss
        elif loss == 'dice':
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
            self.loss = loss

        self.T_MAX = T_MAX

        self.iou = torchmetrics.JaccardIndex(
            num_classes=out_classes, average="micro", task="binary"
        ).to(self.device)
        self.f1 = torchmetrics.F1Score(
            num_classes=out_classes, average="micro", task="binary"
        ).to(self.device)
        self.precision = torchmetrics.Precision(
            num_classes=out_classes, average="micro", task="binary"
        ).to(self.device)
        self.recall = torchmetrics.Recall(
            num_classes=out_classes, average="micro", task="binary"
        ).to(self.device)

        self.to(device)
        self.writer = SummaryWriter(logdir)
        self.iou_v = 0
        self.f1_v  = 0
        self.p_v   = 0
        self.r_v   = 0

        self.dynamic_pos_weight = dynamic_pos_weight

    def forward(self, batch):
        batch = (batch - self.mean.to(batch.device)) / self.std.to(batch.device)
        mask = self.model(batch)
        return mask

    def shared_step(self, batch):
        imgs, masks = batch

        assert imgs.ndim == 4

        h, w = imgs.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        assert masks.ndim == 4
        assert masks.max() <= 1 and masks.min() >= 0

        logits_mask = self.forward(imgs)
        loss = self.loss_fn(logits_mask, masks.float())

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > self.binary_threshold).type(torch.uint8)  # apply thresholding

        return loss, pred_mask, masks

    def reset_metrics(self):
        self.iou.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

    def calculate_metrics(self, pred_mask, gold_mask):
        self.iou_v = self.iou(pred_mask, gold_mask).item()
        self.f1_v  = self.f1(pred_mask, gold_mask).item()
        self.p_v   = self.precision(pred_mask, gold_mask).item()
        self.r_v   = self.recall(pred_mask, gold_mask).item()

    #Logs metrics to a tensorboard
    def log_metrics(self, stage, loss, batch_no):
        self.writer.add_scalar(f'Loss/{stage}', loss, batch_no)
        self.writer.add_scalar(f'IoU/{stage}', self.iou_v, batch_no)

    def update_pos_weight(self, masks):
        pos_count  = masks.sum()
        neg_count  = masks.numel() - pos_count
        pos_weight = neg_count / (pos_count + 1e-8)
        pos_weight = torch.tensor([pos_weight], device=self.device)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    def train_model(
        self, train_loader: DataLoader, val_loader: DataLoader, num_epochs, lr
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_MAX, eta_min=1e-5
        )

        for epoch in range(num_epochs):
            ############
            # TRAINING #
            ############

            self.train()
            train_losses = []
            self.reset_metrics()

            train_progress = tqdm(train_loader,
                                  desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                                  leave=False)
            i = 0
            for batch in train_progress:
                i += 1
                images, masks = batch[0].to(self.device), batch[1].to(self.device)
                if self.dynamic_pos_weight and self.loss == 'bce':
                    self.update_pos_weight(masks)

                optimizer.zero_grad()
                loss, pred_mask, true_mask = self.shared_step((images, masks))

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                self.calculate_metrics(pred_mask, true_mask)

                train_progress.set_postfix(loss=loss.item(), iou=self.iou_v)
                if i % 100 == 0:
                    self.log_metrics('Train', loss, epoch*len(train_loader) + i)

            scheduler.step()

            train_loss_avg = sum(train_losses) / len(train_losses)
            self.log_metrics('Train_epoch', train_loss_avg, epoch)

            ##############
            # VALIDATION #
            ##############
            self.validate_model(val_loader, epoch, num_epochs)

    def validate_model(self, val_loader: DataLoader, epoch, num_epochs):
        self.eval()
        val_losses = []
        self.reset_metrics()

        with torch.no_grad():
            val_progress = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
            )
            i = 0
            for batch in val_progress:
                i += 1
                images, masks = batch[0].to(self.device), batch[1].to(self.device)

                loss, pred_mask, true_mask = self.shared_step((images, masks))
                val_losses.append(loss.item())

                self.calculate_metrics(pred_mask, true_mask)
                val_progress.set_postfix(loss=loss.item())
                if i % 100 == 0:
                    self.log_metrics('Val', loss, epoch*len(val_loader) + i)

        val_loss_avg = sum(val_losses) / len(val_losses)
        self.log_metrics('Val_epoch', val_loss_avg, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss {val_loss_avg:.4f} - Val IoU {self.iou_v:.4f}")


    def test_model(self, test_loader: DataLoader):
        self.eval()
        test_losses = []
        test_ious   = []

        self.reset_metrics()

        image_gallery      = []
        pred_masks_gallery = []

        gt_masks_gallery   = []

        with torch.no_grad():
            test_progress = tqdm(test_loader, desc="Testing model", leave=False)
            i = 0
            for batch in test_progress:
                i += 1
                images, masks = batch[0].to(self.device), batch[1].to(self.device)

                loss, pred_mask, true_mask = self.shared_step((images, masks))
                test_losses.append(loss.item())

                self.calculate_metrics(pred_mask, true_mask)
                test_ious.append(self.iou_v)
                self.log_metrics('Test', loss, i * len(test_loader) + 1)
                if i <= 5:
                    logits = self.forward(images)
                    pr_masks = logits.sigmoid()
                    pr_masks = torch.clip(pr_masks, 0, 1)

                    image_gallery.append(images[0].cpu())
                    gt_masks_gallery.append(masks[0].squeeze().cpu())
                    pred_masks_gallery.append(pr_masks[0].squeeze().cpu())

        test_loss_mean = sum(test_losses) / len(test_losses)
        test_iou_mean  = sum(test_ious) / len(test_ious)
        print(f"Test Loss: {test_loss_mean:.4f}\nTest IoU: {test_iou_mean:.4f}")

        for i,(img, gt, pred) in enumerate(zip(image_gallery, gt_masks_gallery, pred_masks_gallery)):

            fig = self.imshow(image=img.type(torch.uint8),
                        ground_truth=gt,
                        prediction=pred)

            self.writer.add_figure(f'Test/Showcase_{i}', fig, global_step=i)
            plt.close(fig)

    def imshow(title= None, **images):
        """Displays images in one row"""
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(n*4,5))

        for i, (name, image) in enumerate(images.items()):
            ax = axes[i]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(" ".join(name.split("_")).title())
            if name=='image':
                ax.imshow(image.permute(1, 2, 0))
            elif name == 'prediction':
                im = ax.imshow(image)
                # Add color bar next to the prediction image
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks([0, 1])  # Set ticks corresponding to pixel intensities
                cbar.set_ticklabels(['Background', 'Foreground'])  # Set labels for the ticks
                ticks = [i * 0.1 for i in range(11)]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])
            else:
                ax.imshow(image, figure=fig)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig



