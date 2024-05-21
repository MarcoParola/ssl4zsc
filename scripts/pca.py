import torch
import hydra
import os
from sklearn.decomposition import PCA


from src.utils import load_features


@hydra.main(config_path='../config', config_name='config')
def main(cfg):

    features_path = os.path.join(cfg.currentDir, cfg.dataset.path, cfg.model + '_' + cfg.dataset.name)

    features, labels = load_features(features_path)
    print(f'Features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')

    pca = PCA(n_components=3)
    pca.fit(features)
    reduced_features = pca.transform(features)

    print(f'Reduced features shape: {reduced_features.shape}')

    import matplotlib.pyplot as plt
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=labels)
    plt.show()

    # plot tsne
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    reduced_features = tsne.fit_transform(features)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels)
    plt.show()


if __name__ == '__main__':
    main()