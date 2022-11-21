from utils import models_v2 as models, train, datasets

import pytorch_lightning as pl
import torch.nn as nn
import torch


import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#HIDDEN_LSTM = 2048 * 8 * 8
HIDDEN_LSTM = 12288
HIDDEN_SIZE = 64

experiments_name = "road_accidents"


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        # Pick a pretrained model and load the pretrained weights
        model_name = "slow_r50"
        #self.extractor = torch.hub.load('facebookresearch/pytorchvideo:main', 'slow_r50', pretrained=True)
        self.extractor = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
        # 5 for full cnn part
        self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))
        #self.fc1 = nn.Linear(HIDDEN_LSTM, HIDDEN_SIZE)
        models.fix_seeds(33)                
        self.rnn = nn.GRU(input_size=HIDDEN_LSTM,
                           hidden_size=HIDDEN_SIZE,
                           num_layers=1,
                           batch_first=True,
                           dropout=0.5)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)
        #self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        #self.fc2 = nn.Linear(HIDDEN_SIZE // 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):

        #batch_size, C, timesteps, H, W = x.size()
        #c_in = x
        #c_out = self.extractor(c_in)
        r_in = x.flatten(2) # batch_size, timesteps, C*H*W
        r_out, _ = self.rnn(r_in)
        r_out = self.fc(r_out)
        #print(r_out.squeeze())
        #r_out = self.dropout(self.relu(self.fc1(r_out)))
        #r_out = self.dropout(self.fc2(r_out))
        out = torch.sigmoid(r_out)
        return out

bce_base_model = Combine()

for param in bce_base_model.extractor.parameters():
    param.requires_grad = False

train_dataset, test_dataset = datasets.CPDDatasets(
    experiments_name=experiments_name).get_dataset_()

bce_model = models.CPD_model(
                             model=bce_base_model,
                             args=dict(lr=0.001,
                             batch_size=8),
                             num_workers=2,
                             train_dataset=train_dataset,
                             test_dataset=test_dataset)

_ = train.train_model(model=bce_model, max_epochs=1, experiments_name='road_accidents',
                      patience=10, gpus=1, seed=33)