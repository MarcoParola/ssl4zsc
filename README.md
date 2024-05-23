# **ssl4zsc**: Self-Supervised Learning for Zero-Shot Clustering  

Repository for DL experiments on self-supervised learning for generating image embeddings and clustering evaluation.

<img src="https://github.com/MarcoParola/ssl4zsc/assets/32603898/2f8828da-437d-4e3b-a6e6-393de7627dcc" width="450">

<img src="https://github.com/MarcoParola/ssl4zsc/assets/32603898/74fb8575-d0b4-477e-9653-339ef17a0814" width="450">


**Supported models**:
- Convolutional Autoencoder (baseline)
- Variational Convolutional Autoencoder
- [DINO](https://arxiv.org/abs/2104.14294)
- [Vision Transformer Masked Autoencoder ViTMAE](https://arxiv.org/abs/2111.06377)

**Supported datasets**:
- [MNIST](https://ieeexplore.ieee.org/document/726791)
- [Fashion-MNIST](https://arxiv.org/abs/1708.07747)
- CIFAR10
- CIFAR100
- Caltech101
- [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## Install

To install the project, simply clone the repository and get the necessary dependencies:
```sh
git clone https://github.com/MarcoParola/ssl4zsc.git
cd ssl4zsc
```

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```bash
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt
```

Next, create a new project on [Weights & Biases](https://wandb.ai/site) named `ssl4zsc`. Edit `entity` parameter in [config.yaml](https://github.com/MarcoParola/ssl4zsc/blob/main/config/config.yaml) by setting your wandb nick. Log in and paste your API key when prompted.
```sh
wandb login 
```

## Usage

The experiments are organized in a three-step workflow:
- Training SS architectures
- Extract features using a previously trained SS model
- Run clustering algorithms and evaluate cluster results

Additional utility python scripts can be found in `./scripts/`. 
The main ones are:
- `generate_txt_embeddings.py` to precompute the text class description embeddings.
- `pca.py` to plot visualize embeddings in a reduced space (PCA and t-SNE)


### 1. Training
A pretrained model fine-tuning can be run using `train.py` and specifying:
- `model` param from the following list: `cae`, `vcae`, `dino`, `vitmae`
- `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
 

```sh
python train.py model=vcae dataset.name=mnist
```

### 2. Exctract features
After training a model, you can reload it and use it to extract features using `extract_features.py`. 
Features are saved in a directory called `./data/{model param}_{dataset name}/`
Specify the following params:
- `model` param from the following list: `cae`, `vcae`, `dino`, `vitmae`
- `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
- `checkpoint` param by choosing among the pretrained model checkpoints in the output folder (only if model is `cae` or `vcae`). 


```sh
python extract_features.py dataset.name=cifar10 checkpoint=outputs\2024-05-21\11-28-22\lightning_logs\8315z0fs\checkpoints\epoch\=19-step\=25000.ckpt
```

### 3. Clustering and evaluation
After extracting features, you can reload it and use it to cluster data using `clustering.py`. 
Specify the following parameters:
- `model` param from the following list: `cae`, `vcae`, `dino`, `vitmae`
- `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
