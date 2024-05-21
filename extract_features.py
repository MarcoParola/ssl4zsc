import hydra
import torch
import pytorch_lightning as pl
import os
import tqdm

from src.utils import load_dataset, get_early_stopping, get_save_model_callback, get_model
from src.datasets.datamodule import ZeroShotDataModule
from src.log import get_loggers


@hydra.main(config_path='config', config_name='config')
def main(cfg):

    # loggers = get_loggers(cfg)
    loggers = None

    # instantiate the model and load the weights
    model = get_model(cfg)
    model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
    model.load_state_dict(torch.load(model_path)['state_dict'])

    # load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)

    # DataModule
    datamodule = ZeroShotDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=1 # batch size is 1 to extract features one by one
    )
    dataloader = datamodule.test_dataloader() # get test dataloader from datamodule

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        logger=loggers,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )
    #trainer.test(model, dataloader)

    device = torch.device(cfg.train.device)
    model.eval()
    model.to(device)

    # save the features: current directory + data directory + subdirectory (model name _ dataset name) 
    features_path = os.path.join(cfg.currentDir, cfg.dataset.path, cfg.model + '_' + cfg.dataset.name)
    os.makedirs(features_path, exist_ok=True)

    # for with enumerate and tqdm
    for j, batch in enumerate(tqdm.tqdm(dataloader)):
        images, labels, meta = batch
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            # get the features
            features = model.extract_features(images)
            features = features.squeeze()
            actual = meta['label']

            # save the features and labels
            torch.save(features, os.path.join(features_path, f'features_{j}.pt'))
            torch.save(actual, os.path.join(features_path, f'labels_{j}.pt'))
        

if __name__ == '__main__':
    main()