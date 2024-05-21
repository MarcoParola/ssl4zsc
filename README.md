# **ssl4zsc**: Self-Supervised Learning for Zero-Shot Clustering  

Repository for some DL experiments about  

The main actions you can do are:
- train a DL model using `train.py` script
- extract features using a previously trained model
- cluster features previously extracted and evaluate  

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


### 1. Training
A pretrained model fine-tuning can be run using `train.py` and specifying:
- `model` param from the following list: `cae` TODO future architectures could be clip or a transformer-based model
- `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
 

```sh
python train.py dataset.name=cifar10
```

### 2. Exctract features
After trained a model, you can reload it and use it to extract features by using `extract_features.py`. 
Features are saved in a directory called `./data/{model param}_{dataset name}/`
Specify the following params:
- `model` param from the following list: `cae`
- `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
- `checkpoint` param by choosing among the pretrained model checkpoints in the output folder. 



```sh
python extract_features.py dataset.name=cifar10 checkpoint=outputs\2024-05-21\11-28-22\lightning_logs\8315z0fs\checkpoints\epoch\=19-step\=25000.ckpt
```

### 3. Clustering and evaluation
