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

# 0 sequence include cross; 1 sequence no cross
def gen_train_test(image_dir, pos_dir, label_dir, phase, occl_thres):
    num_bbox = 0
    clip_size = 30
    os.system('mkdir -p ../JAAD_processed_data')
    if phase == 'base':
        num_c = 0
        num_nc = 0
    for dir_ in sorted(os.listdir(image_dir)):
        dir_ = dir_.split('.')[0]
        ped_appear = []
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
            peds = [ped for ped in ped_data["objLists"][idx-1] if ped['occl'][0] <= occl_thres]
            num_bbox += len(peds)
            for ped in peds:
                ped['pos'][0] = int(ped['pos'][0]+0.5)
                ped['pos'][1] = int(ped['pos'][1]+0.5)
                ped['pos'][2] = int(ped['pos'][2]+0.5)
                ped['pos'][3] = int(ped['pos'][3]+0.5)               
                ped_dict[ped['id'][0]] = 1
            peds_list[idx] = peds
            output_data[idx] = {'img':img_path, 'pos':[], 'idx':idx, \
                         'dir':dir_}
        if len(ped_dict) == 0:
            print(dir_)
            continue
        ped_appear = {key_:-100 for key_ in ped_dict}
        ped_leave = {key_:-100 for key_ in ped_dict}
        ped_have_leave = {key_:False for key_ in ped_dict}
        ped_cross_start = {key_:-100 for key_ in ped_dict}
        ped_cross_end = {key_:-100 for key_ in ped_dict}
        ped_cross_have_end = {key_:False for key_ in ped_dict}
        ped_width = []
        for i in range(len(peds_list)):
            ped_width.append({})
            peds = peds_list[i+1]
            j = 0
            for ped_id in sorted(ped_dict.keys()):
                have_ped = False
                for ped in peds:
                    if ped_id == ped['id'][0]:
                        have_ped = True
                        if ped_appear[ped_id] == -100:
                            ped_appear[ped_id] = i
                        ped_width[-1][ped_id] = ped['pos'][2]
                        ped_leave[ped_id] = i
                if not have_ped:
                    ped_width[-1][ped_id] = -1
                    if not ped_have_leave[ped_id] and ped_leave[ped_id] != -100:
                        ped_leave[ped_id] = i
                        ped_have_leave[ped_id] = True
                if j < ped_labels.shape[0]:
                    c_label = ped_labels[j,i]
                else:
                    c_label = 0.0
                if c_label == 0:
                    if (not ped_cross_have_end[ped_id]) and ped_cross_end[ped_id] != -100:
                        ped_cross_end[ped_id] = i
                        ped_cross_have_end[ped_id] = True
                else:
                    if ped_cross_start[ped_id] == -100:
                        ped_cross_start[ped_id] = i
                    ped_cross_end[ped_id] = i
                j += 1
        '''
        with open('../JAAD_processed_data/'+dir_+'/appear_cross_list.txt', 'w') as f:
            for ped_id in sorted(ped_dict.keys()):
                f.write(str(ped_appear[ped_id]))
                f.write(' ')
                f.write(str(ped_leave[ped_id]))
                f.write(' ')
                f.write(str(ped_cross_start[ped_id]))
                f.write(' ')
                f.write(str(ped_cross_end[ped_id]))
                f.write('\n')
        '''
        if phase == 'train':
            with open('../JAAD_processed_data/'+dir_+'/train_list_clip_{:d}.txt'.format(clip_size), 'w') as f:
                if clip_size == -1:
                    j = 0
                    for ped_id in sorted(ped_dict.keys()):
                        f.write(str(ped_appear[ped_id]))
                        f.write(' ')
                        f.write(str(ped_leave[ped_id]))
                        f.write(' ')
                        f.write(str(j))
                        if ped_cross_start[ped_id] == -100:
                            f.write(' 1')
                        else:
                            f.write(' 0')
                        f.write('\n')
                        j += 1
                else:
                    j = 0
                    for ped_id in sorted(ped_dict.keys()):
                        if ped_leave[ped_id] - ped_appear[ped_id] >= clip_size:
                            for k in range(ped_appear[ped_id], ped_leave[ped_id]-clip_size+1):
                                f.write(str(k))
                                f.write(' ')
                                f.write(str(k+clip_size))
                                f.write(' ')
                                f.write(str(j))
                                if ped_cross_start[ped_id] == -100 or ped_cross_end[ped_id] <= k or ped_cross_start[ped_id] >= k+clip_size:
                                    f.write(' 1')
                                elif k <= ped_cross_start[ped_id] and (k+clip_size) >= ped_cross_end[ped_id]:
                                    f.write(' 0')
                                elif k >= ped_cross_start[ped_id] and (k+clip_size) <= ped_cross_end[ped_id]:
                                    f.write(' 2')
                                elif k <= ped_cross_start[ped_id] and (k+clip_size) <= ped_cross_end[ped_id]:
                                    f.write(' 3')
                                elif k >= ped_cross_start[ped_id] and (k+clip_size) >= ped_cross_end[ped_id]:
                                    f.write(' 4')
                                f.write('\n')
                        j += 1   
        if phase == 'base':
            with open('../JAAD_processed_data/'+dir_+'/list_clip_{:d}_base.txt'.format(clip_size), 'w') as f:
                    j = 0
                    for ped_id in sorted(ped_dict.keys()):
                        if ped_leave[ped_id] - ped_appear[ped_id] >= clip_size and ped_cross_start[ped_id] > 0:
                            for k in range(ped_appear[ped_id], ped_leave[ped_id]-clip_size+1):
                              ped_is_good = True
                              for s in range(clip_size):
                                if ped_width[k+s][ped_id] < 60:
                                    ped_is_good = False
                              if ped_is_good:
                                f.write(str(k))
                                f.write(' ')
                                f.write(str(k+clip_size))
                                f.write(' ')
                                f.write(str(j))
                                if ped_cross_end[ped_id] <= k+clip_size or ped_cross_start[ped_id] >= k+clip_size:
                                #if ped_cross_start[ped_id] >= k+clip_size:
                                    f.write(' 0')
                                    num_nc += 1
                                elif ped_cross_start[ped_id] < k+clip_size <= ped_cross_end[ped_id]:
                                    f.write(' 1')
                                    num_c += 1
                                f.write('\n')
                        j += 1   

        else:
            with open('../JAAD_processed_data/'+dir_+'/test_list.txt', 'w') as f:
                j = 0
                for ped_id in sorted(ped_dict.keys()):
                    if ped_cross_start[ped_id] != -100:
                        f.write(str(ped_appear[ped_id]))
                        f.write(' ')
                        f.write(str(ped_leave[ped_id]))
                        f.write(' ')
                        f.write(str(j))
                        f.write('\n')
                    j += 1
    if phase == 'base':
        print(num_nc)
        print(num_c)
    print(num_bbox)
gen_train_test('/data/JAAD/JAAD_clip_images', '../JAAD_vbb/vbb_part', '../JAAD_cross_label', 'base', 0)
