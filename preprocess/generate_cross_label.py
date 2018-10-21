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


def func(image_dir, pos_dir, label_dir, occl_thres):
    os.system('mkdir -p ../JAAD_processed_data')
    for dir_ in sorted(os.listdir(image_dir)):
        dir_ = dir_.split('.')[0]
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
            idx = int(img_id.split(".")[0])
            peds = [ped for ped in ped_data["objLists"][idx-1] if ped['occl'][0] < occl_thres]
            for ped in peds:
                ped_dict[ped['id'][0]] = 1
            peds_list[idx] = peds
            output_data[idx] = {'idx':idx, \
                         'dir':dir_}
        if len(ped_dict) == 0:
            print(dir_)
            continue
        for i in range(len(peds_list)):
            if ped_labels.shape[0] > 0:  
                output_data[i+1]['labels'] = ped_labels[:,i]
            else:
                output_data[i+1]['labels'] = np.array([])
        os.system('mkdir -p ../JAAD_processed_data/'+dir_+'/cross_labels')
        for key_ in sorted(output_data.keys()):
            np.save('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_ + '/cross_labels/{:03d}.npy'.format(output_data[key_]['idx']), output_data[key_]['labels'])
func('/data/JAAD_clip_images', '../JAAD_vbb/vbb_full', '../JAAD_cross_label', 100)
