import pytorch_lightning as pl
import torch
import detectron2
from detectron2.modeling import build_model
from detectron2.config import get_cfg

class BarcodeSegmentationModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images = list(image.to(self.device) for image in batch)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images = list(image.to(self.device) for image in batch)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("val_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.BASE_LR)
        return optimizer
