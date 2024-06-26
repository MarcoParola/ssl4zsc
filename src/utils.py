import torch
import torchvision
import numpy as np
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.cae import Autoencoder
from src.models.vcae import VariationalAutoencoder
from src.models.dino import Dino
from src.models.vitmae import VitMAE


def get_save_model_callback(save_path):
    """Returns a ModelCheckpoint callback
    cfg: hydra config
    """
    save_model_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=save_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
    )
    return save_model_callback

def get_early_stopping(patience=10):
    """Returns an EarlyStopping callback
    cfg: hydra config
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
    )
    return early_stopping_callback

def load_dataset(dataset, data_dir, resize=256, val_split=0.2, test_split=0.2):

    train, val, test = None, None, None

    torch.manual_seed(42)
    np.random.seed(42)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((resize, resize)),
    ])

    # CIFAR-10
    if dataset == 'cifar10':
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # CIFAR-100
    elif dataset == 'cifar100':
        train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # Caltech101
    elif dataset == 'caltech101':
        data = torchvision.datasets.Caltech101(data_dir, download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split+test_split:], indices[:val_split], indices[val_split:val_split+test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)
    # ImageNet
    elif dataset == 'imagenet':
        from torchvision.datasets import ImageFolder
        val = datasets.load_dataset('mrm8488/ImageNet1K-val', split='train')

        class ImageNetDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform=None):
                self.dataset = dataset
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, lbl = self.dataset[idx]['image'], self.dataset[idx]['label']
                if self.transform:
                    img = self.transform(img)
                # check if the image contains 1 channels
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                return img, lbl

        val = ImageNetDataset(val, transform)
        train = val
        test = val
        

    # Oxford-IIIT Pet
    elif dataset == 'oxford-iiit-pet':
        data = torchvision.datasets.OxfordIIITPet(data_dir, download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split+test_split:], indices[:val_split], indices[val_split:val_split+test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)

    # SVHN
    elif dataset == 'svhn':
        #data = torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=transform)
        train = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=transform)
        test = torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # MNIST
    elif dataset == 'mnist':
        train = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # FashionMNIST
    elif dataset == 'fashionmnist':
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return train, val, test


def load_features(data_dir):
    features, labels = None, None

    features = []
    labels = []
    for file in os.listdir(data_dir):
        if file.startswith('features'):
            features.append(torch.load(os.path.join(data_dir, file)).cpu().numpy())
            label = file.split('-')[1].split('.')[0]
            labels.append(int(label))
        #elif file.startswith('labels'):
        #    labels.append(torch.load(os.path.join(data_dir, file)).cpu().numpy())

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def get_model(cfg):
    model_name = cfg.model
    model = None
    if model_name == 'cae':
        model = Autoencoder(
            ae_params=cfg.ae,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )
    elif model_name == 'vcae':
        model = VariationalAutoencoder(
            ae_params=cfg.ae,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )
    elif model_name == 'dino':
        model = Dino()
    elif model_name == 'vitmae':
        model = VitMAE()

    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    return model

    

if __name__ == "__main__":

    data = [
        # 'cifar10',
        # 'cifar100',
        # 'caltech101',
        'imagenet',
        # 'oxford-iiit-pet',
        # 'svhn',
        # 'mnist',
        # 'fashionmnist',
    ]
    for dataset in data:
        print(f'\n\nDataset: {dataset}')
        data = load_dataset(dataset, './data')
        print(data[0].__len__(), data[1].__len__(), data[2].__len__())

        test = data[2]
        print(test)
        import matplotlib.pyplot as plt
        for i in range(10):
            img, lbl = test.__getitem__(i)
            print(img.shape, lbl)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(lbl)
            plt.show()

        # import matplotlib.pyplot as plt
        # dataloader = torch.utils.data.DataLoader(data[2], batch_size=4, shuffle=True)
        # for i, batch in enumerate(dataloader):
        #     for j in range(4):
        #         img, lbl = batch[0][j], batch[1][j]
        #         print(img.shape, lbl)
        #         plt.imshow(img.permute(1, 2, 0))
        #         plt.title(lbl)
        #         plt.show()

        '''
        for d in data:
            print(f'\nData: {d}')

            for i in range(len(d)):
                _, label = d[i]
                if label not in class_distribution:
                    class_distribution[label] = 0
                class_distribution[label] += 1
            
            # sort and print the class distribution
            class_distribution = dict(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True))
            # for key, value in class_distribution.items():
            #     print(f'{key}: {value}')

            # print number of classes
            print(f'Number of classes: {len(class_distribution)}')

            # compute the class unbalance as the ratio between the number of samples in the most frequent class and the number of samples in the least frequent class
            dist = list(class_distribution.values())
            class_unbalance = max(dist) / min(dist)
            print(f'Class unbalance: {class_unbalance}')
        '''