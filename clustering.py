import torch
import hydra
import os
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from src.utils import load_features


@hydra.main(config_path='./config', config_name='config')
def main(cfg):

    features_path = os.path.join(cfg.currentDir, cfg.dataset.path, cfg.model + '_' + cfg.dataset.name)
    features, labels = load_features(features_path)
    
    # min-max normalization column by column
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - features[:, i].min()) / (features[:, i].max() - features[:, i].min())

    print(f'Features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')

    # perform clustering
    n_clusters = cfg[cfg.dataset.name].n_classes
    #clustering_alg = KMeans(n_clusters=n_clusters, random_state=0)
    clustering_alg = HDBSCAN(min_cluster_size=50)
    clusters = clustering_alg.fit(features)

    #print(clusters.cluster_centers_)
    #print(len(clusters.labels_))

    # compute adjusted mutual information and silhouette score evaluation metrics
    print(f'Adjusted Mutual Information: {adjusted_mutual_info_score(labels, clusters.labels_)}')
    print(f'Silhouette Score: {silhouette_score(features, clusters.labels_)}')


if __name__ == '__main__':
    main()