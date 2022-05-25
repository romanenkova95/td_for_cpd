from utils import datasets, kl_cpd, models, metrics

import warnings

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

import numpy as np
import time
from sklearn import manifold
import matplotlib.pyplot as plt

experiments_name = "explosion"
train_dataset, test_dataset = datasets.CPDDatasets(
    experiments_name=experiments_name
).get_dataset_()

args = {}
args["wnd_dim"] = 4
args["RNN_hid_dim"] = 16
args["batch_size"] = 4
args["lr"] = 1e-4
args["weight_decay"] = 0.0
args["grad_clip"] = 10
args["CRITIC_ITERS"] = 5
args["weight_clip"] = 0.1
args["lambda_ae"] = 0.1  # 0.001
args["lambda_real"] = 10  # 0.1
args["num_layers"] = 1
# args['data_dim'] = 12288
args["data_dim"] = 2400
# args["data_dim"] = 600
# args["data_dim"] = 6
# args['data_dim'] = 48
args["emb_dim"] = 100

args["in_channels"] = 3

args["window_1"] = 4
args["window_2"] = 4

args["sqdist"] = 50


seed = 112
models.fix_seeds(seed)
experiments_name = "explosion"

netG = models.NetG(args)
netD = models.NetD(args)
netE = models.NetE(args)

kl_cpd_model = models.KLCPDVideo(
    netG,
    netD,
    args,
    extractor=netE,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
)

logger = TensorBoardLogger(save_dir="logs/explosion", name="kl_cpd")
early_stop_callback = EarlyStopping(
    monitor="val_mmd2_real_D",
    stopping_threshold=1e-5,
    verbose=True,
    mode="min",
    patience=5,
)


# for param in kl_cpd_model.extractor.parameters():
#     param.requires_grad = False

trainer = pl.Trainer(
    max_epochs=100,
    gpus="1",
    benchmark=True,
    check_val_every_n_epoch=1,
    gradient_clip_val=args["grad_clip"],
    logger=logger,
    callbacks=early_stop_callback,
)

trainer.fit(kl_cpd_model)

threshold_number = 25
threshold_list = np.linspace(-5, 5, threshold_number)
threshold_list = 1 / (1 + np.exp(-threshold_list))
threshold_list = [-0.001] + list(threshold_list) + [1.001]

_, delay_list, fp_delay_list = metrics.evaluation_pipeline(
    kl_cpd_model,
    kl_cpd_model.val_dataloader(),
    threshold_list,
    device="cuda",
    model_type="klcpd",
    verbose=False,
)


path_to_saves = "saves/"
metrics.write_metrics_to_file(
    path_to_saves + "result_metrics_custom_conv2d_win4_600.txt", _, ""
)

plt.figure(figsize=(12, 12))
plt.plot(fp_delay_list.values(), delay_list.values(), "-o", markersize=8, label="TSCP")
plt.xlabel("Mean Time to False Alarm", fontsize=28)
plt.ylabel("Mean Detection Delay", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc="upper left", fontsize=26)
plt.savefig(path_to_saves + "_custom_conv2d_win4_600" + "curve.png")
