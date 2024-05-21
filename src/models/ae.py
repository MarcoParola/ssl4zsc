import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
import numpy as np
import os
import hydra

'''
class AutoencoderModule(LightningModule):

    def __init__(self, ae_param, img_size, lr=10e-6, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.loss = torch.nn.MSELoss()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, ae_param.conv1, kernel_size=3, stride=2, padding=1),
            torch.nn.MaxPool2d(kernel_size=ae_param.pool, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ae_param.conv1, ae_param.conv2, kernel_size=3, stride=2, padding=1),
            torch.nn.MaxPool2d(kernel_size=ae_param.pool, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ae_param.conv2, ae_param.conv3, kernel_size=3, stride=2, padding=1),
            torch.nn.MaxPool2d(kernel_size=ae_param.pool, stride=2),
            torch.nn.ReLU())
        self.latent_dim = ae_param.latent_dim

        self.fc1 = torch.nn.Linear(128, self.latent_dim)
        self.fc2 = torch.nn.Linear(self.latent_dim, 128)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ae_param.conv3, ae_param.conv2, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Upsample(scale_factor=ae_param.pool, mode='bilinear', align_corners=True),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(ae_param.conv2, ae_param.conv1, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Upsample(scale_factor=ae_param.pool, mode='bilinear', align_corners=True),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(ae_param.conv1, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Upsample(scale_factor=ae_param.pool, mode='bilinear', align_corners=True),
            torch.nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        shape_before_flattening = x.shape
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(shape_before_flattening)
        x = self.decoder(x)
        return x

    def extract_features(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        imgs, labels, meta = batch
        y_hat = self(imgs)
        loss = self.loss(y_hat, imgs)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss
        
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.eval()
        img, label, meta = batch
        return self(img)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        imgs, labels, meta = batch
        y_hat = self(imgs)
        loss = self.loss(y_hat, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        return loss

    
@hydra.main(config_path='../../config', config_name='config')
def main(cfg):
    print(cfg.ae)
    model = AutoencoderModule(cfg.ae, cfg.dataset.resize, cfg.train.lr, cfg.train.max_epochs)
    img_size = cfg.dataset.resize

    img = torch.randn(2, 3, img_size, img_size)
    out = model(img)
    print(img.shape, out.shape)



if __name__ == '__main__':
    main()
'''






# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


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
            nn.Tanh()
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

    '''
    def save_images(self, x, output, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """

        if self.hparams.batch_size < n:
            raise IndexError("You are trying to plot more images than your batch contains!")

        # denormalize images
        denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
        x = [denormalization(i) for i in x[:n]]
        output = [denormalization(i) for i in output[:n]]

        # make grids and save to logger
        grid_top = vutils.make_grid(x, nrow=n)
        grid_bottom = vutils.make_grid(output, nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid)
    '''

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "train_input_output")

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"avg_val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "test_input_output")

        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"avg_test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": logs}


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



