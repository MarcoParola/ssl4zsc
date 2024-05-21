import torch

class ZeroShotDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, idx):
        img, lbl = self.orig_dataset.__getitem__(idx)
        if self.transform:
            img = self.transform(img)

        meta = {
            'label': lbl,
            # add description of the class
        }

        return img, img, meta





if __name__ == "__main__":
    import torchvision
    from matplotlib import pyplot as plt
    from src.utils import load_dataset
    train, val, test = load_dataset('cifar10', 'data', 224)
    dataset = ZeroShotDataset(train)
    image, label, meta = dataset.__getitem__(0)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()
    plt.imshow(label.permute(1, 2, 0).numpy())
    plt.show()
    print(meta)



