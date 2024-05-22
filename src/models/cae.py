import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
import numpy as np
import os
import hydra


class Autoencoder(pl.LightningModule):
    def __init__(self, ae_params, lr=10e-6, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()

 
        self.ae_params = ae_params
        self.lr = lr
        self.max_epochs = max_epochs


        self.encoder = nn.Sequential(
            # input (nc) x 128 x 128
            nn.Conv2d(ae_params.n_channels, ae_params.enc_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps),
            nn.LeakyReLU(True),
            # input (nfe) x 64 x 64
            nn.Conv2d(ae_params.enc_feature_maps, ae_params.enc_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 2),
            nn.LeakyReLU(True),
            # input (nfe*2) x 32 x 32
            nn.Conv2d(ae_params.enc_feature_maps * 2, ae_params.enc_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 4),
            nn.LeakyReLU(True),
            # input (nfe*4) x 16 x 16
            nn.Conv2d(ae_params.enc_feature_maps * 4, ae_params.enc_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 8),
            nn.LeakyReLU(True),
            # input (nfe*8) x 8 x 8
            nn.Conv2d(ae_params.enc_feature_maps * 8, ae_params.enc_feature_maps * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 16),
            nn.LeakyReLU(True),
            # input (nfe*16) x 4 x 4
            nn.Conv2d(ae_params.enc_feature_maps * 16, ae_params.latent_dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ae_params.latent_dim),
            nn.LeakyReLU(True)
            # output (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            nn.ConvTranspose2d(ae_params.latent_dim, ae_params.dec_feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 16),
            nn.ReLU(True),
            # input (nfd*16) x 4 x 4
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 16, ae_params.dec_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 8),
            nn.ReLU(True),
            # input (nfd*8) x 8 x 8
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 8, ae_params.dec_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 4),
            nn.ReLU(True),
            # input (nfd*4) x 16 x 16
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 4, ae_params.dec_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 2),
            nn.ReLU(True),
            # input (nfd*2) x 32 x 32
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 2, ae_params.dec_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps),
            nn.ReLU(True),
            # input (nfd) x 64 x 64
            nn.ConvTranspose2d(ae_params.dec_feature_maps, ae_params.n_channels, 4, 2, 1, bias=False),
            # sigmoid instead of tanh for stability
            nn.Sigmoid()
            #nn.Tanh()
            # output (nc) x 128 x 128
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def extract_features(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        return self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch, batch_idx, stage):
        imgs, labels, meta = batch
        y_hat = self(imgs)
        loss = F.mse_loss(y_hat, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        return loss




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



