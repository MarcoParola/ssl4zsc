import torch
import hydra
import os
from sklearn.decomposition import PCA


from src.utils import load_features


@hydra.main(config_path='../config', config_name='config')
def main(cfg):

    features_path = os.path.join(cfg.currentDir, cfg.dataset.path, cfg.model + '_' + cfg.dataset.name)

    features, labels = load_features(features_path)
    # min-max normalization column by column
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - features[:, i].min()) / (features[:, i].max() - features[:, i].min())

    print(f'Features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')

    pca = PCA(n_components=2)
    pca.fit(features)
    reduced_features = pca.transform(features)

    print(f'Reduced features shape: {reduced_features.shape}')

    import matplotlib.pyplot as plt
    cbar = plt.colorbar(plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, s=5, alpha=0.5))
    cbar.set_label('Labels')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA ' + cfg.dataset.name + ' ' + cfg.model)
    plt.show()

    # plot tsne
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    reduced_features = tsne.fit_transform(features)
    cbar = plt.colorbar(plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, s=5, alpha=0.5))
    cbar.set_label('Labels')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.title('TSNE ' + cfg.dataset.name + ' ' + cfg.model)
    plt.show()


if __name__ == '__main__':
    main()