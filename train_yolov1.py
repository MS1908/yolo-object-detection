import torch
import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics import detection

from models import YOLOv1
from losses import YOLOv1Loss
from dataset import kaggle_car_loader_factory_v1


class YOLOv1Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = YOLOv1(num_classes=1)
        self.map = detection.mean_ap.MeanAveragePrecision(box_format='cxcywh')

    def forward(self, x):
        outputs = self.net(x)
        return outputs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters, lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_pred = self.net(X)

        criterion = YOLOv1Loss(num_classes=1)
        loss = criterion(y_pred, y)
