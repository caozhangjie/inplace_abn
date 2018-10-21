from itertools import chain

import glob
import torch
from PIL import Image
from os import path
from torch.utils.data import Dataset
import os
import os.path as opath
import numpy as np
import pdb
import random


def func(image_dir, pos_dir, label_dir, phase, occl_thres):
    clip_size = 14
    os.system('mkdir -p ../JAAD_processed_data')
    for dir_ in sorted(os.listdir(image_dir)):
        dir_ = dir_.split('.')[0]
        #if (int(dir_.split('_')[1]) > 250 and phase == 'train') or \
        #   (int(dir_.split('_')[1]) <= 250 and phase == 'test'):
        #    continue
        ped_data = np.load(os.path.join(pos_dir, dir_+'.npy')).item()
        num_frames = ped_data['objLists']
        ped_dict = {}
        peds_list = {}
        ped_labels = []
        output_data = {}
        label_list = os.listdir(opath.join(label_dir, dir_))
        labels_list = []
        if "pedestrian.npy" in label_list:
            labels_list.append(opath.join(label_dir, dir_, "pedestrian.npy"))
        for i in range(13):
            if "pedestrian"+str(i)+".npy" in label_list:
                labels_list.append(opath.join(label_dir, dir_, "pedestrian"+str(i)+".npy"))
        ped_label_list = []
        for f_name in labels_list:
            ped_label_list.append(np.load(f_name))
        ped_labels = np.array(ped_label_list)

        for img_id in os.listdir(path.join(image_dir, dir_+'.mp4')):
            img_path = path.join(image_dir, dir_+'.mp4', img_id)
            idx = int(img_id.split(".")[0])
            peds = [ped for ped in ped_data["objLists"][idx-1] if ped['occl'][0] < occl_thres]
            for ped in peds:
                ped['pos'][0] = int(ped['pos'][0]+0.5)
                ped['pos'][1] = int(ped['pos'][1]+0.5)
                ped['pos'][2] = int(ped['pos'][2]+0.5)
                ped['pos'][3] = int(ped['pos'][3]+0.5)
                #ped['pos'][2] = ped['pos'][0] + ped['pos'][2] - 1
                #ped['pos'][3] = ped['pos'][1] + ped['pos'][3] - 1
                ped_dict[ped['id'][0]] = 1
            peds_list[idx] = peds
            output_data[idx] = {'img':img_path, 'pos':[], 'idx':idx, \
                         'dir':dir_}
        if len(ped_dict) == 0:
            print(dir_)
            continue
        for i in range(len(peds_list)):
            peds = peds_list[i+1]
            j = 0
            for ped_id in sorted(ped_dict.keys()):
                have_ped = False
                for ped in peds:
                    if ped_id == ped['id'][0]:
                        have_ped = True
                        output_data[i+1]['pos'].append(ped['pos'])
                if not have_ped:
                    output_data[i+1]['pos'].append(np.array([-1,-1,-1,-1], dtype=np.int32))
                j += 1
            output_data[i+1]['pos'] = np.array(output_data[i+1]['pos'], dtype=np.int32)
            if ped_labels.shape[0] > 0:  
                output_data[i+1]['labels'] = ped_labels[:,i,:]
            else:
                output_data[i+1]['labels'] = np.array([])
        os.system('mkdir -p ../JAAD_processed_data/'+dir_+'/pos')
        os.system('mkdir -p ../JAAD_processed_data/'+dir_+'/labels')
        with open('../JAAD_processed_data/'+dir_+'/lists.txt', 'w') as f:
            for key_ in sorted(output_data.keys()):
                f.write(output_data[key_]['img']+' '+'/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_ + '/pos/{:03d}.npy'.format(output_data[key_]['idx'])+' '+'/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_ + '/labels/{:03d}.npy'.format(output_data[key_]['idx'])+"\n")
                np.save('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_ + '/pos/{:03d}.npy'.format(output_data[key_]['idx']), output_data[key_]['pos'])
                np.save('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_ + '/labels/{:03d}.npy'.format(output_data[key_]['idx']), output_data[key_]['labels'])

        '''
        ped_num = output_data_pos.size(0) - 64
        ped_appear = [[] for i in range(ped_num)]
        pos_list = []
        feature_list = []
        cross_ped_num = ped_labels.shape[0]
        cross_start = [[] for i in range(cross_ped_num)]
        cross_end = [[] for i in range(cross_ped_num)]
        for i in range(num_frames):               
            pos = torch.load(opath.join(pos_dir, dir_, '{:d}.pt'.format(i+1)))
            for j in range(ped_num):
                if pos[j+64, 0] >= 0:
                    ped_appear[j].append(i)
            pos_list.append(pos)
            feature_list.append(opath.join(image_dir, dir_+'.mp4', '{:d}.pt'.format(i+1)))
            for j in range(cross_ped_num):
                if ped_labels[j][i] == 1 and (i == 0 or ped_labels[j][i-1] == 0):
                    cross_start[j].append(i)
                if ped_labels[j][i] == 1 and (i == ped_labels.shape[1]-1 or ped_labels[j][i+1] == 0):
                    cross_end[j].append(i)
        valid_ped_list = []
        for j in range(ped_num):
            ped_appear[j] = [ped_appear[j][0], ped_appear[j][-1]]
            if ped_appear[j][1] + 1 - ped_appear[j][0] >= self.clip_size:
                valid_ped_list.append(j)
        cross_ped_list = []
        for i in range(cross_ped_num):
            if cross_start[i] and cross_start[i][0] < ped_appear[i][0]:
                cross_start[i][0] = ped_appear[i][0]
            if cross_end[i] and cross_end[i][0] > ped_appear[i][1]:
                cross_end[i][0] = ped_appear[i][1]
            if cross_start[i] and (ped_appear[i][1] - ped_appear[i][0] + 1 >= self.clip_size):
                cross_ped_list.append(i)
        if len(cross_ped_list) > 0:
            self.cross_index.append(int(dir_.split('_')[1]))
        self.videos[int(dir_.split('_')[1])] = [pos_list, feature_list, ped_labels, cross_ped_list, ped_appear]
        '''
func('/data/JAAD_clip_images', '../JAAD_vbb/vbb_full', '/data/JAAD_behavioral_encode', 'test', 100)
