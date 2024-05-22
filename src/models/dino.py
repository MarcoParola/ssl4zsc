import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
import numpy as np
import os
import hydra


class Dino(pl.LightningModule):
    def __init__(self, lr=10e-6, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.max_epochs = max_epochs
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)


    def forward(self, x):
        x = self.model(x)
        return x

    def extract_features(self, x):
        return self.model(x)




@hydra.main(config_path='../../config', config_name='config')
def main(cfg):

    model = Autoencoder(cfg.ae, cfg.train.lr, cfg.train.max_epochs)

    # define random input
    x = torch.randn(2, 3, cfg.dataset.resize, cfg.dataset.resize)

    print('input shape:', x.shape)
    print('output shape:', model(x).shape)
    print('encoding shape:', model.extract_features(x).shape)

if __name__ == '__main__':
    main()



