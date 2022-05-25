# KL-CPD for Video

KL-CPD, a novel kernel learning framework for time series CPD that optimizes a lower bound of test power via an auxiliary generative model. In this project, we try to improve performance of this method in terms of quality and efficiency to make it applicable to video survelliance.

## Installation

You can find list of necessary packages in `environment.yml`.

## KL-CPD for video.

The training and test process for main model is placed in `kl-cpd-example-with-test.ipynb` jupyter notebook.

## Experiments with Model Compression

### Tensor Layer.

To train KL-CPD model with linear layer in TCL (tensor contraction layer) format, use following command 
```python3 train_tl.py --block-type tcl3d  --bias-rank N```
where N is integer or "full", defalt is 4. 

To train KL-CPD model with linear layer in TRL (tensor regression layer) format, use following command 
```python3 train_tl.py --block-type trl```

### Pruning.
To train KL-CPD model with pruning, use following command 
```python3 train_prune.py -q Q```
where Q denotes prune ratio, default is 0.5.

### Model Performance Testing

To evaluate model, use command 
```python3 test_tl.py TIMESTAMP -tn 25```
where TIMESTAMP can be found in model checkpoint name.


## Experiments with preprocessing
To train model with custom CNN run `kl-cpd-preprocessing.ipynb` notebook. To run experiments with resizing, run `iterate_compressions.py`


## Some Result

### Pruning

Model | Embedding size | Hidden size | \# param, M | F1 | AUC
:-:|:-:|:-:|:-:|:-:|:-: 
Original | 100 | 16 | 2.46 | 0.3333 | 1.8753
Pruned   |  50 | 16 | 1.23 | 0.3000 | 1.0511
Pruned   |  10 | 16 | 0.25 | 0.2807 | 1.0866

### Low-Rank Format

Model | Embedding size | Hidden size | Bias rank | \# param, M | FLOPs, M | F1 | AUC
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
Original | 100     |  16        | full | 4     | 81.8  | 0.3333 | 1.8753
TLC | (32, 8, 8) | (16, 4, 4) | 1    | 0.236 | 58.0  | 0.3636 | 0.7900
TLC | (32, 8, 8) | (16, 4, 4) | 2    | 0.237 | 58.0  | 0.4848 | 0.6785
TLC | (32, 8, 8) | (16, 4, 4) | 4    | 0.240 | 58.0  | 0.4118 | 0.7458
TLC | (32, 8, 8) | (16, 4, 4) | 8    | 0.244 | 58.0  | 0.4444 | 0.8158
TLC | (32, 8, 8) | (16, 4, 4) | full | 0.280 | 58.0  | 0.5333    | 0.6804
TLC | (64, 8, 8) | (32, 4, 4) | 8    | 0.407 | 114.5 | 0.5714 | 0.6322
TLC | (64, 8, 8) | (32, 4, 4) | full    | 0.459 | 114.5 | 0.4348 | 0.7095
TRC | (32, 8, 8) | (16, 4, 4) | 0    | 0.892 | - | 0.5263 | 0.7817