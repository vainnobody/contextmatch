# ContextMatch

This repository contains the official implementation of the ContextMatch.

## Getting Started


### Environment
Create a new environment and install the requirements:
```shell
pip install -r requirements.txt
```


### Pretrained Backbone:
[ResNet-101]
```bash
mkdir pretrained
```
Please put the pretrained model under `pretrained` dictionary.


## Usage

### Training ContextMatch

```bash
sh scripts/train.sh <num_gpu> <port>
```
To run on different labeled data partitions or different datasets, please modify:

``config``, ``labeled_id_path``, ``unlabeled_id_path``, and ``save_path`` in [train.sh]

### Evaluation
```bash
sh scripts/inference.sh <num_gpu> <port>
```
To evaluate your checkpoint, please modify ``checkpoint_path`` in [inference.sh]
