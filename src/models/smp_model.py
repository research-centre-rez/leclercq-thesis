import torch
import segmentation_models_pytorch as smp
import torchmetrics
from torch.optim import lr_scheduler
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm


class SMP_model(nn.Module):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        out_classes: int,
        T_MAX: int,
        threshold,
        **kwargs
    ):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )

        self.binary_threshold = threshold

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.T_MAX = T_MAX

        self.iou       = torchmetrics.JaccardIndex(num_classes=out_classes,
                                                   average='micro',
                                                   task='binary').to('cuda')
        self.f1        = torchmetrics.F1Score(num_classes=out_classes,
                                              average='micro',
                                              task='binary').to('cuda')
        self.precision = torchmetrics.Precision(num_classes=out_classes,
                                                average='micro',
                                                task='binary').to('cuda')
        self.recall    = torchmetrics.Recall(num_classes=out_classes,
                                             average='micro',
                                             task='binary').to('cuda')

    def forward(self, batch):
        batch = (batch - self.mean.to(batch.device)) / self.std.to(batch.device)
        mask = self.model(batch)
        return mask

    def shared_step(self, batch):
        imgs, masks = batch

        assert imgs.ndim == 4

        h,w = imgs.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert masks.ndim == 4
        assert masks.max() <=1 and masks.min() >= 0

#        for img, mask in batch:
#            logits_mask = self.forward(img)
#            loss = self.loss_fn(logits_mask, mask)

        logits_mask = self.forward(imgs)
        loss        = self.loss_fn(logits_mask, masks)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > self.binary_threshold).type(torch.uint8) #apply thresholding

        return loss, pred_mask, masks

    def reset_metrics(self):
        self.iou.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

    def calculate_metrics(self, pred_mask, gold_mask):
        self.iou(pred_mask, gold_mask)
        self.f1(pred_mask, gold_mask)
        self.precision(pred_mask, gold_mask)
        self.recall(pred_mask, gold_mask)

    def log_metrics(self, stage):
        iou_val  = self.iou.compute().item()
        f1_val   = self.f1.compute().item()
        prec_val = self.precision.compute().item()
        rec_val  = self.recall.compute().item()

        print(f'{stage} Metrics: IoU: {iou_val:.4f}, F1: {f1_val:.4f}, Precision: {prec_val:.4f}, Recall: {rec_val:.4f}')
        return {
            'iou':iou_val,
            'f1':f1_val,
            'precision':prec_val,
            'recall':rec_val,
        }
    
    def train_model(self, train_loader:DataLoader, val_loader:DataLoader, num_epochs, lr):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_MAX, eta_min=1e-5)

        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            self.reset_metrics()

            train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
            for batch in train_progress:
                images, masks = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()
                loss, pred_mask, true_mask = self.shared_step((images, masks))

                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())

                self.calculate_metrics(pred_mask, true_mask)

                iou_val = self.iou.compute().item()
                train_progress.set_postfix(loss=loss.item(), iou=iou_val)

            scheduler.step()

            train_loss_avg = sum(train_losses) / len(train_losses)
            train_metrics  = self.log_metrics('Train')

            print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss {train_loss_avg:.4f}')

            #Validate
            self.eval()
            val_losses = []
            self.reset_metrics()

            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
                for batch in val_progress:
                    images, masks = batch[0].to(device), batch[1].to(device)

                    loss, pred_mask, true_mask = self.shared_step((images, masks))
                    val_losses.append(loss.item())

                    self.calculate_metrics(pred_mask, true_mask)
                    val_progress.set_postfix(loss=loss.item())

            val_loss_avg = sum(val_losses) / len(val_losses)
            val_metrics  = self.log_metrics('Validation')

            print(f'Epoch {epoch + 1}/{num_epochs} - Val Loss {val_loss_avg:.4f}')

    def test_model(self, test_loader:DataLoader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        self.eval()
        test_losses = []

        self.reset_metrics()

        with torch.no_grad():
            test_progress = tqdm(test_loader, desc=f'Testing model')
            for batch in test_progress:
                images, masks = batch[0].to(device), batch[1].to(device)

                loss, pred_mask, true_mask = self.shared_step((images, masks))
                test_losses.append(loss.item())

                self.calculate_metrics(pred_mask, true_mask)

        test_loss_mean = sum(test_losses) / len(test_losses)
        test_metrics = self.log_metrics('Test')

        print(f'Test Loss: {test_loss_mean:.4f}')


















