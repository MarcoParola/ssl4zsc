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
A pretrained model fine-tuning can be run using `train.py` and specifying:
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
 

```sh
python train.py dataset.name=cifar10
```

After trained a model, you can reload it and use it to extract features by using `extract_features.py`. Specify the following params:
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
- the `checkpoint` param by choosing among the pretrained model checkpoints in the output folder. Pleas note, in the following example the `checkpoint` param is valued according the windows path format.



```sh
python extract_features.py dataset.name=cifar10 checkpoint=...
```

