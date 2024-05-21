import hydra
import torch
import pytorch_lightning as pl
import os

from src.utils import load_dataset, get_early_stopping, get_save_model_callback
from src.datasets.datamodule import ZeroShotDataModule
from src.models.ae import Autoencoder
from src.log import get_loggers



@hydra.main(config_path='config', config_name='config')
def main(cfg):

    # Set seed
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)

    # callback
    callbacks = list()
    callbacks.append(get_early_stopping(cfg.train.patience))
    model_save_dir = os.path.join(cfg.currentDir, cfg.train.save_path)
    callbacks.append(get_save_model_callback(model_save_dir))

    # loggers
    loggers = get_loggers(cfg)

    # Load dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    
    # DataModule
    datamodule = ZeroShotDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=cfg.train.batch_size
    )
    

    model = Autoencoder(
        ae_params=cfg.ae,
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        logger=loggers,
        log_every_n_steps=cfg.train.log_every_n_steps,
        #deterministic=True
    )

    trainer.fit(model, datamodule=datamodule)

    # predict first batch and save predicted images
    imgs, labels, meta = next(iter(datamodule.test_dataloader()))
    y_hat = model(imgs)
    from matplotlib import pyplot as plt
    for i in range(len(imgs)):
        # plot both original and predicted images in the same figure
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(imgs[i].permute(1, 2, 0))
        ax[0].set_title("Original")
        ax[0].axis("off")
        ax[1].imshow(y_hat[i].detach().permute(1, 2, 0))
        ax[1].set_title("Reconstructed")
        ax[1].axis("off")
        plt.savefig(f"reconstructed_{i}.png")

    # Test
    trainer.test(model, datamodule=datamodule)
    

    




if __name__ == '__main__':
    main()