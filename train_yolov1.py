import argparse
import torch
import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics import detection

from models import YOLOv1Tiny
from losses import YOLOv1Loss
from dataset import kaggle_car_loader_factory_v1

criterion = YOLOv1Loss(num_classes=1)


class YOLOv1Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = YOLOv1Tiny(num_classes=1)

    def forward(self, x):
        outputs = self.net(x)
        return outputs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_pred = self.net(X)
        loss = criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.net(X)
        loss = criterion(y_pred, y)
        self.log('valid_loss', loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='./data/')
    args = parser.parse_args()

    model = YOLOv1Model()
    trainer = pl.Trainer()
    train_loader, val_loader = kaggle_car_loader_factory_v1(root=args.train_data, img_h=448, img_w=448, bs=1)
    trainer.fit(model, train_loader, val_loader)
