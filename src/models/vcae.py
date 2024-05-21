import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
import numpy as np
import os
import hydra


from torch.distributions import Normal

class VariationalAutoencoder(pl.LightningModule):
    def __init__(self, ae_params, lr=1e-5, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()

        self.ae_params = ae_params
        self.lr = lr
        self.max_epochs = max_epochs

        self.encoder = nn.Sequential(
            nn.Conv2d(ae_params.n_channels, ae_params.enc_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps),
            nn.LeakyReLU(True),
            nn.Conv2d(ae_params.enc_feature_maps, ae_params.enc_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 2),
            nn.LeakyReLU(True),
            nn.Conv2d(ae_params.enc_feature_maps * 2, ae_params.enc_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 4),
            nn.LeakyReLU(True),
            nn.Conv2d(ae_params.enc_feature_maps * 4, ae_params.enc_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 8),
            nn.LeakyReLU(True),
            nn.Conv2d(ae_params.enc_feature_maps * 8, ae_params.enc_feature_maps * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.enc_feature_maps * 16),
            nn.LeakyReLU(True),
            nn.Conv2d(ae_params.enc_feature_maps * 16, ae_params.latent_dim * 2, 4, 1, 0, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ae_params.latent_dim, ae_params.dec_feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 16, ae_params.dec_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 8, ae_params.dec_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 4, ae_params.dec_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ae_params.dec_feature_maps * 2, ae_params.dec_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ae_params.dec_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(ae_params.dec_feature_maps, ae_params.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.view(z.size(0), self.ae_params.latent_dim, 1, 1)
        return self.decoder(z)

    # method used in the commont step for computing the loss as it returns. reconstructed image, mu and log_var
    def __forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def forward(self, x):
        img, mu, log_var = self.__forward(x)
        return img

    def extract_features(self, x):
        mu, _ = self.encode(x)
        return mu

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
        imgs_reconstructed, mu, log_var = self.__forward(imgs)
        recon_loss = F.mse_loss(imgs_reconstructed, imgs, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        return loss




@hydra.main(config_path='../../config', config_name='config')
def main(cfg):

    model = VariationalAutoencoder(cfg.ae, cfg.train.lr, cfg.train.max_epochs)

    # define random input
    x = torch.randn(2, 3, cfg.dataset.resize, cfg.dataset.resize)

    img, mu, log_var = model(x)

    print('input shape:', x.shape)
    print('output shape:', img.shape)
    print('encoding shape:', model.extract_features(x).shape)

if __name__ == '__main__':
    main()



