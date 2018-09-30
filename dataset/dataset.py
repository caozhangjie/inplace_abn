from itertools import chain

import glob
import torch
from PIL import Image
from os import path
from torch.utils.data import Dataset
import os
import os.path as opath

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

class JAADCrossDataset(Dataset):
    def __init__(self, feature_dir, pos_dir, label_dir):
        super(JAADCrossDataset, self).__init__()
        self.videos = []
        for dir_ in sorted(os.listdir(label_dir)):
            self.videos.append([])
            ped_list = os.listdir(opath.join(label_dir, dir_))
            peds_list = []
            if "pedestrian.npy" in ped_list:
                peds_list.append(opath.join(label_dir, dir_, "pedestrian.npy"))
            for i in range(13):
                if "pedestrian"+str(i)+".npy" in ped_list:
                    peds_list.append(opath.join(label_dir, dir_, "pedestrian"+str(i)+".npy"))
            ped_label_list = []
            for f_name in peds_list:
                ped_label_list.append(np.load(f_name))
            ped_labels = np.array(ped_label_list)
            for i in range(ped_labels.shape[1]):
                self.videos[-1].append([opath.join(feature_dir, dir_, '{:d}.npy'.format(i+1)), \
                            opath.join(pos_dir, dir_, '{:d}.npy'.format(i+1)), ped_labels[:, i]])

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.videos)
