import torch

class ZeroShotDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, idx):
        img, lbl = self.orig_dataset.__getitem__(idx)
        img = torch.clamp(img, 0, 1) # clamp the image to be between 0 and 1
        # clip image to be three channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
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
    # print max and min values
    print(torch.max(image), torch.min(image))
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()
    plt.imshow(label.permute(1, 2, 0).numpy())
    plt.show()
    print(meta)



