from itertools import chain

import glob
import torch
from PIL import Image
from os import path
from torch.utils.data import Dataset
import os
import numpy as np

class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, img_root, bbox_root, seg_root, feature_root, pos_root, transform, phase, interval):
        super(SegmentationDataset, self).__init__()

        self.transform = transform

        dir_list = os.listdir(img_root)
        self.imgs = []
        for dir_ in dir_list:
            dir_id = int(dir_.split("_")[1].split(".")[0])
            ped_dict = {}
            peds_list = []
            temp = []
            if ((phase == 'test' and dir_id > 250) or (phase == 'train' and dir_id <= 250)) and (interval[0] <= dir_id < interval[1]):
                bbox = np.load(path.join(bbox_root, "video_{:04d}.npy".format(dir_id))).item()
                for img_id in os.listdir(path.join(img_root, dir_)):
                    img_path = path.join(img_root, dir_, img_id)
                    idx = int(img_id.split(".")[0])
                    peds = [ped for ped in bbox["objLists"][idx-1] if ped['occl'][0] < 0.5]
                    for ped in peds:
                        ped['pos'][0] = int(ped['pos'][0]+0.5)
                        ped['pos'][1] = int(ped['pos'][1]+0.5)
                        ped['pos'][2] = int(ped['pos'][2]+0.5)
                        ped['pos'][3] = int(ped['pos'][3]+0.5)
                        #ped['pos'][2] = ped['pos'][0] + ped['pos'][2] - 1
                        #ped['pos'][3] = ped['pos'][1] + ped['pos'][3] - 1
                        ped_dict[ped['id'][0]] = 1
                    peds_list.append(peds)
                    temp.append({'img':img_path, 'pos':[], 'ped_appear':None, \
                    'seg_out':path.join(seg_root, "video_{:04d}".format(dir_id), str(idx)+'.png'), \
                    'feature_out':path.join(feature_root, "video_{:04d}".format(dir_id), str(idx)+'.pt'), \
                    'pos_out':path.join(pos_root, "video_{:04d}".format(dir_id), str(idx)+'.pt'), 'idx':idx, 'dir':dir_id})
                if len(ped_dict) == 0:
                    print(ped_dict)
                    continue
                for i in range(len(peds_list)):
                    peds = peds_list[i]
                    temp[i]['ped_appear'] = np.zeros([len(ped_dict)])
                    j = 0
                    for ped_id in sorted(ped_dict.keys()):
                        have_ped = False
                        for ped in peds:
                            if ped_id == ped['id'][0]:
                                have_ped = True
                                temp[i]['ped_appear'][j] = 1.0
                                temp[i]['pos'].append(ped['pos'])
                        if not have_ped:
                            temp[i]['pos'].append(np.array([-1,-1,-1,-1], dtype=np.int32))
                        j += 1
                    temp[i]['pos'] = np.array(temp[i]['pos'], dtype=np.int32)
                self.imgs += temp

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        with Image.open(self.imgs[index]["img"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))
        item1 = {}
        for key in self.imgs[index]:
            item1[key] = self.imgs[index][key]
        item1['img'] = img
        item1['size'] = size
        item1['in_path'] = self.imgs[index]["img"]
        return item1

def segmentation_collate(items):
    output = {}
    for key in items[0]:
        output[key] = []
    for item in items:
        for key in item:
            output[key].append(item[key])
    output['img'] = torch.stack(output['img'])
    
    return output
