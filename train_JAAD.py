import argparse
from functools import partial
from os import path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim

import models
from dataset.dataset import JAADCrossDataset
from dataset.transform import SegmentationTransform
from models.graph import GraphCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAAD graph model")
    parser.add_argument("--scales", metavar="LIST", type=str, default="[0.7, 1, 1.2]", help="List of scales")
    parser.add_argument('--gpu_id', metavar='GPU_ID', type=int, default=0, help='GPU ID')
    parser.add_argument('--feature_dim', type=int, default=1000, help='input feature dimension')
    parser.add_argument('--test_interval', type=int, default=0, help='test interval')

    args = parser.parse_args()
    feature_dim = args.feature_dim
    graph_dim = 256
    torch.cuda.set_device(args.gpu_id)
    cudnn.benchmark = True
    train_dset = JAADCrossDataset('./JAAD_features', './JAAD_pos', './JAAD_cross_label', "train")
    train_loader = DataLoader(train_dset, batch_size=1, pin_memory=True, num_workers=4, shuffle=True)
    test_dset = JAADCrossDataset('./JAAD_features', './JAAD_pos', './JAAD_cross_label', "test")
    test_loader = DataLoader(test_dset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)

    graph_model = GraphCNN(feature_dim, [1024, 512, graph_dim])
    graph_model = graph_model.cuda()
    sequence_model = nn.LSTM(input_size=graph_dim, hidden_size=256, num_layers=1, batch_first=True, dropout=0.5)
    sequence_model = sequence_model.cuda()
    cross_classifier = nn.Linear(256, 2)
    cross_classifier = cross_classifier.cuda()
    
    optimizer = optim..Adam([{"params":graph_model.parameters()}, \
                             {"params":sequence_model.parameters()}, \
                             {"params":cross_classifier.parameters()}], lr=0.001)
    
    for epoch in range(args.epoch):
        graph_model.train()
        sequence_model.train()
        for batch_id, data in enumerate(train_loader):
            optimizer.zero_grad()
            fea, pos, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
            seq_len = fea.size(1)
            fea = fea.view(-1, feature_dim)
            graph_fea = graph_model(fea).view(-1, seq_len, graph_dim)
            ped_fea = graph_fea[64:, :, :]
            loss = 0
            ped_out = sequence_model(ped_fea)
            ped_prob = nn.Softmax(dim=1)(cross_classifier(ped_out.view(-1, ped_out.size(2))))
            loss = nn.CrossEntropyLoss(ped_prob, label.view(-1))
            loss.backward()
            optimizer.step()
            log_str = "Epoch: {:04d}, Iter: {:05d}, Loss: {:.4f}".format(epoch, batch_id, loss.item())
            if (epoch * len(fea_loader) + batch_id) % args.test_interval == 0:
                for batch_id, data in enumerate(test_loader):
                    pass
