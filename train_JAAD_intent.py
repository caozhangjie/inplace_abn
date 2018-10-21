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
from dataset.dataset import JAADIntentDataset, JAADCollateIntent
from torch.utils.data import DataLoader
from models.graph import init_weights

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

    train_dset = JAADIntentDataset('train', clip_size=14)
    train_loader = DataLoader(train_dset, batch_size=1, pin_memory=True, num_workers=4, shuffle=True, collate_fn=JAADCollateIntent)
    test_dset = JAADIntentDataset("test")
    test_loader = DataLoader(test_dset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False, collate_fn=JAADCollateIntent)

    sequence_model = nn.LSTM(input_size=feature_dim, hidden_size=256, num_layers=1, batch_first=False)
    sequence_model = sequence_model.cuda()
    pos_embedding = nn.Sequential(nn.Linear(4, feature_dim//2), nn.ReLU())
    pos_embedding = pos_embedding.cuda()
    global_embedding = nn.Sequential(nn.Linear(feature_dim//2, 256), nn.ReLU())
    global_emebdding = global_embedding.cuda()
    cross_classifier = nn.Linear(512, 2)
    cross_classifier = cross_classifier.cuda()
    init_weights(cross_classifier)
    init_weights(sequence_model)
    init_weights(pos_embedding)
    init_weights(global_embedding)
    
    optimizer = optim.Adam([{"params":sequence_model.parameters()}, \
                             {"params":cross_classifier.parameters()}], lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.epoch):
        sequence_model.train()
        cross_classifier.train()
        for batch_id, data in enumerate(train_loader):
            step_lr_scheduler(optimizer, epoch * len(train_loader) + batch_id, 500, 0.3)
            optimizer.zero_grad()
            features, pos, label, c_label, i_label, img_sizes = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[4].cuda(), data[5].cuda()
            label = label.long()
            c_label = c_label.long()
            i_label = i_label.long()
            img_sizes = img_sizes.float()
            pos = pos.float()
            pos[:,0] = (pos[:,0] / img_sizes[:,0] - 0.5) * 10.0
            pos[:,2] = (pos[:,2] / img_sizes[:,0] - 0.5) * 10.0
            pos[:,1] = (pos[:,1] / img_sizes[:,1] - 0.5) * 10.0
            pos[:,3] = (pos[:,3] / img_sizes[:,1] - 0.5) * 10.0
            seq_len = features.size(0)
            fea_dim = features.size(1) // 2
            global_feature = features[:, 0:fea_dim]
            ped_feature = features[:, fea_dim:]
            pos_feature = pos_embedding(pos)
            input_features = torch.cat((ped_feature, pos_feature), dim=1)
            loss = 0
            ped_out = sequence_model(input_features.view(seq_len, 1, feature_dim))[0]
            ped_prob = cross_classifier(torch.cat( (ped_out.view(-1, ped_out.size(2)), \
                                                    global_embedding(global_feature)), dim=1))
            loss = loss_func(ped_prob, c_label.view(-1))
            loss.backward()
            optimizer.step()
            log_str = "Epoch: {:04d}, Iter: {:05d}, Loss: {:.4f}".format(epoch, batch_id, loss.item())
            #print(log_str)
            if (epoch * len(train_loader) + batch_id) % args.test_interval == 0:
                #all_positive = 0
                #all_negative = 0
                sequence_model.eval()
                cross_classifier.eval()
                pos_embedding.eval()
                avg_acc = 0
                num_test = 0
                for test_id, data in enumerate(test_loader):
                    features, pos, label, c_label, i_label, img_sizes = data[0].cuda(), data[1].cuda(), data[2], data[3], data[4], data[5].cuda()
                    label = label.long()
                    c_label = c_label.long()
                    i_label = i_label.long()
                    img_sizes = img_sizes.float()
                    pos = pos.float()
                    pos[:,0] = (pos[:,0] / img_sizes[:,0] - 0.5) * 10.0
                    pos[:,2] = (pos[:,2] / img_sizes[:,0] - 0.5) * 10.0
                    pos[:,1] = (pos[:,1] / img_sizes[:,1] - 0.5) * 10.0
                    pos[:,3] = (pos[:,3] / img_sizes[:,1] - 0.5) * 10.0
                    seq_len = features.size(0)
                    fea_dim = features.size(1) // 2
                    global_feature = features[:, 0:fea_dim]
                    ped_feature = features[:, fea_dim:]
                    pos_feature = pos_embedding(pos)
                    input_features = torch.cat((ped_feature, pos_feature), dim=1)
                    ped_out = sequence_model(input_features.view(seq_len, 1, feature_dim))[0]
                    ped_prob = cross_classifier(torch.cat( (ped_out.view(-1, ped_out.size(2)), \
                                                    global_embedding(global_feature)), dim=1))
                    _, ped_pred = torch.max(ped_prob, 1)
                    #all_positive += torch.sum(c_label==1)
                    #all_negative += torch.sum(c_label==0)
                    avg_acc += float(torch.sum(ped_pred.cpu()==c_label))
                    num_test += seq_len
                if args.debug:
                    pdb.set_trace()
                #print(all_positive)
                #print(all_negative)
                print("Epoch: {:04d}, Iter: {:05d}, accuracy: {:.4f}".format(epoch, batch_id, avg_acc/num_test))
                sequence_model.train()
                cross_classifier.train()
