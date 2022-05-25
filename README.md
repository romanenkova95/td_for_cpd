# KL-CPD for Video

## Installation

You can find list of necessary packages in `environment.yml`.

## KL-CPD for video.

The training and test process for main model is placed in `kl-cpd-example-with-test.ipynb` jupyter notebook.

## Experiments with Tensor Layer.

To tran kl-cpd model with linear layer in TCL (tensor contraction layer) format, launch `train_tl.py` script.
To evaluate model with TCL layers, launch `test_tl.py`.

## Experiments with preprocessing
To train model with custom CNN run `kl-cpd-preprocessing.ipynb` notebook. To run experiments with resizing, run `iterate_compressions.py`
