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
from torchvision import transforms
from PIL import Image
import pycocotools.mask as mask_util
import pickle

def calc_angle(p1, p2):
    eps = 0.0000001
    vec = p1 - p2
    if vec[0] > 0:
        vec = -vec
    vec_len = np.sqrt(np.abs(vec[0]*vec[0]+vec[1]*vec[1]))
    angle = np.arccos(vec[1] / (vec_len+eps))
    return angle

def triangle(p1, p2, p3):
    eps = 0.0000001
    v12 = p2 - p1
    v13 = p3 - p1
    v12_len = np.sqrt(np.abs(v12[0]*v12[0]+v12[1]*v12[1]))
    v13_len = np.sqrt(np.abs(v13[0]*v13[0]+v13[1]*v13[1]))
    return np.arccos(np.sum(v12 * v13) / ((v12_len * v13_len) + eps))

def calc_triangle(p1, p2, p3):
    return triangle(p1, p2, p3), triangle(p2, p1, p3), triangle(p3, p1, p2)

def calc_distance(p1, p2):
    dx = np.abs(p1[0]-p2[0])
    dy = np.abs(p1[1]-p2[1])
    return dx, dy, np.sqrt(dx*dx+dy*dy)

# keypoint connections [1, 2], [1, 0], [2, 0], [2, 4], [1, 3], [6, 8], [8, 10], [5, 7], [7, 9], [12, 14], [14, 16], [11, 13], [13, 15], [6, 5], [12, 11]
def keypoint_feature(keyps, edges):
    triangles = [[0, 1, 2], [0,2,4], [1,2,4], [1,2,3],[6,8,10], [5,7,9], [12,14,16], [11,13,15], [11,12,13], [11,12,14], [5,6,7], [5,6,8]]
    _, _, length = calc_distance(keyps[5,:], keyps[16,:])
    features = []
    for edge in edges:
        dx, dy, dis = calc_distance(keyps[edge[0], :], keyps[edge[1], :])
        dx /= length
        dy /= length
        dis /= length
        features.append(dx)
        features.append(dy)
        features.append(dis)
        features.append(calc_angle(keyps[edge[0], :], keyps[edge[1], :])/np.pi)
    for triangle in triangles:
        angle1, angle2, angle3 = calc_triangle(keyps[triangle[0], :], keyps[triangle[1], :], keyps[triangle[2], :])
        features.append(angle1/np.pi)
        features.append(angle2/np.pi)
        features.append(angle3/np.pi)
    return np.array(features)

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

def train_transform(resize_size=256, crop_size=224):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def test_transform(resize_size=256, crop_size=224):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, threshold, transform):
        super(SegmentationDataset, self).__init__()

        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self.images = []
        for sub_dir in sorted(os.listdir(self.in_dir)):
         if (threshold - 30) <= int(sub_dir.split(".")[0].split("_")[1]) < threshold:
          for img_path in chain(*(glob.iglob(path.join(self.in_dir, sub_dir, ext)) for ext in SegmentationDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            self.images.append({
                "idx": sub_dir.split(".")[0] + "_" + idx,
                "path": img_path
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))

        return {"img": img, "meta": {"idx": self.images[item]["idx"], "size": size}}


def segmentation_collate(items):
    imgs = torch.stack([item["img"] for item in items])
    metas = [item["meta"] for item in items]

    return {"img": imgs, "meta": metas}

class JAADIntentDataset(Dataset):
    def __init__(self, phase, clip_size=-1):
        super(JAADIntentDataset, self).__init__()
        self.phase = phase
        self.videos = {}
        self.info = {"size":{}, 'densepose':{}}
        for dir_ in sorted(os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data')):
            temp_dict = open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/list1.txt').readlines()
            id_list = [[int(line.split()[0].split('/')[-1].split('.')[0]), line.split()] for line in temp_dict]
            self.videos[int(dir_.split('_')[1])] = {}
            for val in id_list:
                self.videos[int(dir_.split('_')[1])][val[0]] = val[1]
                self.info['size'][int(dir_.split('_')[1])] = [[int(size_line.split()[0]), int(size_line.split()[1])] for size_line in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/size_list.txt').readlines()]
        if self.phase == 'train':
            self.train_list = []
            for dir_ in os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'):
                if int(dir_.split('_')[1]) <= 250:
                    temp_list = [val.split() for val in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/train_list_clip_{:d}.txt'.format(clip_size), 'r').readlines()]
                    temp_list = [[int(num) for num in val] + [int(dir_.split('_')[1])] for val in temp_list]
                    self.train_list += temp_list
            self.train_dict = {}
            for entries in self.train_list:
                if entries[3] not in self.train_dict:
                    self.train_dict[entries[3]] = []
                self.train_dict[entries[3]].append(entries)
        elif self.phase == 'test':
            self.test_list = []
            for dir_ in os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'):
                if int(dir_.split('_')[1]) > 250:
                    temp_list = [val.split() for val in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/test_list.txt', 'r').readlines()]
                    temp_list = [[int(num) for num in val] + [int(dir_.split('_')[1])] for val in temp_list]
                    self.test_list += temp_list
                

    def __getitem__(self, index):
      if self.phase == 'train':
        data_cls = sorted(self.train_dict.keys())[int(random.random() * 100000000) % len(self.train_dict)]
        s_frame, e_frame, ped_id, data_cls_new, video_index = self.train_dict[data_cls][int(random.random() * 100000000) % len(self.train_dict[data_cls])]
        assert data_cls_new == data_cls
      elif self.phase == 'test':
        s_frame, e_frame, ped_id, video_index = self.test_list[index]
      all_features = []
      all_ped_pos = []
      all_ped_label = []
      all_ped_c_label = []
      all_ped_i_label = []
      all_img_size = []
      for i in range(s_frame, e_frame):
          feature_path, pos_path, label_path, c_label_path, i_label_path, densepose_path \
             = self.videos[video_index][i+1]
          size_img = self.info['size'][video_index][i]
          all_img_size.append(size_img)
          ## description feature
          feature = torch.load(feature_path)
          global_feature = feature[0, :]
          local_feature = feature[ped_id+1, :]
          ## position
          ped_pos = np.load(pos_path)[ped_id, :]          
          ## behavior label / cross label / intent label
          labels = np.load(label_path)
          c_labels = np.load(c_label_path)
          i_labels = np.load(i_label_path)
          ## densepose feature
          densepose = pickle.load(open(densepose_path, 'rb'), encoding='latin1')
          masks = mask_util.decode(densepose['masks'])
          keyps = densepose['keyps']
          boxes = densepose['boxes']
          bodys = densepose['bodys']
          if ped_id < labels.shape[0]:
              ped_label = labels[ped_id, :]
              ped_c_label = c_labels[ped_id]
              ped_i_label = i_labels[ped_id]
          else:
              ped_label = np.zeros([9])
              ped_c_label = 0.0
              ped_i_label = 0.0
          all_ped_pos.append(ped_pos)
          all_ped_label.append(ped_label)
          all_features.append(torch.cat((global_feature, local_feature)))
          all_ped_c_label.append(ped_c_label)
          all_ped_i_label.append(ped_i_label)
      all_ped_pos = np.array(all_ped_pos)
      all_ped_label = np.array(all_ped_label)
      all_ped_c_label = np.array(all_ped_c_label)
      all_ped_i_label = np.array(all_ped_i_label)
      return torch.stack(all_features), torch.from_numpy(all_ped_pos), torch.from_numpy(all_ped_label), torch.from_numpy(all_ped_c_label), torch.from_numpy(all_ped_i_label), torch.from_numpy(np.array(all_img_size))

    def __len__(self):
        if self.phase == 'train':
            return len(self.videos)
        elif self.phase == 'test':
            return len(self.test_list)

def JAADCollateIntent(items):
    #features = [item[0] for item in items]
    #poses = [item[1] for item in items]
    #label = [item[2] for item in items]
    #cross_ped_id = [item[3] for item in items]
    return items[0]

class JAADClassificationDataset(Dataset):
    def __init__(self, phase, clip_size=-1):
        super(JAADClassificationDataset, self).__init__()
        self.phase = phase
        self.videos = {}
        self.info = {"size":{}, 'densepose':{}}
        for dir_ in sorted(os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data')):
            temp_dict = open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/list1.txt').readlines()
            id_list = [[int(line.split()[0].split('/')[-1].split('.')[0]), line.split()] for line in temp_dict]
            self.videos[int(dir_.split('_')[1])] = {}
            for val in id_list:
                self.videos[int(dir_.split('_')[1])][val[0]] = val[1]
                self.info['size'][int(dir_.split('_')[1])] = [[int(size_line.split()[0]), int(size_line.split()[1])] for size_line in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/size_list.txt').readlines()]
        if self.phase == 'train':
            self.train_list = []
            for dir_ in os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'):
                if int(dir_.split('_')[1]) <= 250:
                    temp_list = [val.split() for val in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/list_clip_{:d}_base.txt'.format(clip_size), 'r').readlines()]
                    temp_list = [[int(num) for num in val] + [int(dir_.split('_')[1])] for val in temp_list]
                    self.train_list += temp_list
            self.train_dict = {}
            for entries in self.train_list:
                if entries[3] not in self.train_dict:
                    self.train_dict[entries[3]] = []
                self.train_dict[entries[3]].append(entries)
        elif self.phase == 'test':
            self.test_list = []
            for dir_ in os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'):
                if int(dir_.split('_')[1]) > 250:
                    temp_list = [val.split() for val in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/list_clip_{:d}_base.txt'.format(clip_size), 'r').readlines()]
                    temp_list = [[int(num) for num in val] + [int(dir_.split('_')[1])] for val in temp_list]
                    self.test_list += temp_list
                

    def __getitem__(self, index):
      if self.phase == 'train':
        data_cls = sorted(self.train_dict.keys())[int(random.random() * 100000000) % len(self.train_dict)]
        s_frame, e_frame, ped_id, data_cls_new, video_index = self.train_dict[data_cls][int(random.random() * 100000000) % len(self.train_dict[data_cls])]
        assert data_cls_new == data_cls
      elif self.phase == 'test':
        s_frame, e_frame, ped_id, data_cls, video_index = self.test_list[index]
      all_features = []
      all_ped_pos = []
      all_ped_label = []
      all_img_size = []
      all_keyp_features = []
      for i in range(s_frame, e_frame):
          feature_path, pos_path, label_path, c_label_path, i_label_path, densepose_path \
             = self.videos[video_index][i+1]
          size_img = self.info['size'][video_index][i]
          all_img_size.append(size_img)
          ## image
          
          ## position
          ped_pos = np.load(pos_path)[ped_id, :]
          ## densepose feature
          densepose = pickle.load(open(densepose_path, 'rb'), encoding='latin1')
          masks = mask_util.decode(densepose['masks'])[:,:,ped_id]
          keyps = np.transpose(densepose['keyps'][ped_id][0:2, :])
          #boxes = densepose['boxes']
          bodys = densepose['bodys'][ped_id]
          kp_lines = densepose['kp_lines']
          keyp_features = keypoint_feature(keyps, kp_lines)
          all_keyp_features.append(keyp_features)
          all_ped_pos.append(ped_pos)
          all_features.append(torch.cat((global_feature, local_feature)))
      all_ped_pos = np.array(all_ped_pos)
      return torch.stack(all_features), torch.from_numpy(all_ped_pos), data_cls, torch.from_numpy(np.array(all_img_size)), torch.from_numpy(np.array(all_keyp_features))

    def __len__(self):
        if self.phase == 'train':
            return 2000
        elif self.phase == 'test':
            return len(self.test_list)

def JAADCollateClassification(items):
    features = torch.stack([item[0] for item in items])
    poses = torch.stack([item[1] for item in items])
    label = torch.Tensor([item[2] for item in items])
    size_img = torch.stack([item[3] for item in items])
    keyp_features = torch.stack([item[4] for item in items])
    return features, poses, label, size_img, keyp_features

if __name__ == '__main__':
    train_dset = JAADClassificationDataset('train', clip_size=14)
    pdb.set_trace()
    print(len(train_dset[0]))
