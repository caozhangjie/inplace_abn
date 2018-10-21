import argparse
from os import path
import torch
import torchvision
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import pdb
import models.network as network
from dataset.dataset import JAADClassificationDataset, JAADCollateClassification
from torch.utils.data import DataLoader
from models.graph import init_weights
import pickle

def normalize_pos(pos, img_sizes):
    pos[:,:,0] = (pos[:,:,0] / img_sizes[:,:,0] - 0.5) * 10.0
    pos[:,:,2] = (pos[:,:,2] / img_sizes[:,:,0] - 0.5) * 10.0
    pos[:,:,1] = (pos[:,:,1] / img_sizes[:,:,1] - 0.5) * 10.0
    pos[:,:,3] = (pos[:,:,3] / img_sizes[:,:,1] - 0.5) * 10.0

def step_lr_scheduler(optimizer, iter_num, step_size, gamma):
    if iter_num % step_size == step_size - 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * gamma
    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAAD graph model")
    parser.add_argument('--gpu_id', metavar='GPU_ID', type=int, default=0, help='GPU ID')
    parser.add_argument('--feature_dim', type=int, default=10752, help='input feature dimension')
    parser.add_argument('--test_interval', type=int, default=500, help='test interval')
    parser.add_argument('--epoch', type=int, default=100, help='train epochs')
    parser.add_argument('--debug', action='store_true', help='debug switch')

    args = parser.parse_args()
    feature_dim = args.feature_dim
    torch.cuda.set_device(args.gpu_id)
    cudnn.benchmark = True

    train_dset = JAADClassificationDataset('train', clip_size=14)
    train_loader = DataLoader(train_dset, batch_size=10, pin_memory=True, num_workers=4, shuffle=True, collate_fn=JAADCollateClassification)
    test_dset = JAADClassificationDataset("test", clip_size=14)
    test_loader = DataLoader(test_dset, batch_size=10, pin_memory=True, num_workers=4, shuffle=False, collate_fn=JAADCollateClassification)

    conv_model = network.TemporalConvModel(feature_dim*3, 14)
    conv_model = conv_model.cuda()
    pos_embedding = nn.Sequential(nn.Linear(4, feature_dim), nn.ReLU())
    pos_embedding = pos_embedding.cuda()
    init_weights(conv_model)
    init_weights(pos_embedding)
    
    optimizer = optim.Adam([{"params":conv_model.parameters()}, \
                             {"params":pos_embedding.parameters()}], lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.epoch):
        conv_model.train()
        pos_embedding.train()
        for batch_id, data in enumerate(train_loader):
            step_lr_scheduler(optimizer, epoch * len(train_loader) + batch_id, 500, 0.3)
            optimizer.zero_grad()
            features, pos, label, img_sizes, keyp_features = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[4].cuda()
            label = label.long()
            img_sizes = img_sizes.float()
            pos = pos.float()
            normalize_pos(pos, img_sizes)
            seq_len = features.size(1)
            fea_dim = features.size(2) // 2
            #global_feature = features[:, 0:fea_dim]
            #ped_feature = features[:, fea_dim:]
            print(keyp_features.size())
            pos_feature = pos_embedding(pos.view(-1,4)).view(-1,seq_len,fea_dim)
            input_features = torch.cat((features, pos_feature), dim=2)
            ped_prob = conv_model(input_features.permute(0,2,1).contiguous())
            loss = loss_func(ped_prob, label)
            loss.backward()
            optimizer.step()
            log_str = "Epoch: {:04d}, Iter: {:05d}, Loss: {:.4f}".format(epoch, batch_id, loss.item())
            print(log_str)
            if (epoch * len(train_loader) + batch_id) % args.test_interval == 0:
                conv_model.eval()
                pos_embedding.eval()
                avg_acc = 0
                num_test = 0
                for test_id, data in enumerate(test_loader):
                  if test_id < 100:
                    print('{:d}/{:d}'.format(test_id,len(test_loader)))
                    features, pos, label, img_sizes = data[0].cuda(), data[1].cuda(), data[2], data[3].cuda()
                    label = label.long()
                    img_sizes = img_sizes.float()
                    pos = pos.float()
                    normalize_pos(pos, img_sizes)
                    seq_len = features.size(1)
                    fea_dim = features.size(2) // 2
                    global_feature = features[:, 0:fea_dim]
                    ped_feature = features[:, fea_dim:]
                    pos_feature = pos_embedding(pos.view(-1,4)).view(-1,seq_len,fea_dim)
                    input_features = torch.cat((features, pos_feature), dim=2)
                    ped_prob = conv_model(input_features.permute(0,2,1).contiguous())
                    _, ped_pred = torch.max(ped_prob, 1)
                    avg_acc += float(torch.sum(ped_pred.cpu()==label))
                    num_test += features.size(0)
                if args.debug:
                    pdb.set_trace()
                print("Epoch: {:04d}, Iter: {:05d}, accuracy: {:.4f}".format(epoch, batch_id, avg_acc/num_test))
                conv_model.train()
                pos_embedding.train()
