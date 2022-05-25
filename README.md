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
