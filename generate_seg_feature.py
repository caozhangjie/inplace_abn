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

import models
from dataset.dataset_segment import SegmentationDataset, segmentation_collate
from dataset.transform import SegmentationTransform
from modules.bn import ABN
from modules.deeplab import DeeplabV3
import math
import os

parser = argparse.ArgumentParser(description="Testing script for the Vistas segmentation model")
parser.add_argument("--scales", metavar="LIST", type=str, default="0.7, 1, 1.2", help="List of scales")
parser.add_argument("--flip", action="store_true", help="Use horizontal flipping")
parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
                    help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
                    default="final",
                    help="How the output files are formatted."
                         " -- palette: color coded predictions"
                         " -- raw: gray-scale predictions"
                         " -- prob: gray-scale predictions plus probabilities")
parser.add_argument("snapshot", metavar="SNAPSHOT_FILE", type=str, help="Snapshot file to load")
#parser.add_argument("data", metavar="IN_DIR", type=str, help="Path to dataset")
#parser.add_argument("output", metavar="OUT_DIR", type=str, help="Path to output folder")
parser.add_argument("--world_size", metavar="WS", type=int, default=1, help="Number of GPUs")
parser.add_argument("--rank", metavar="RANK", type=int, default=0, help="GPU id")
parser.add_argument("--start", type=int, default=0, help="start id")
parser.add_argument("--threshold", metavar="THRESHOLD", type=int, default=100000, help="GPU id")



def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class SegmentationModule(nn.Module):
    _IGNORE_INDEX = 255

    class _MeanFusion:
        def __init__(self, x, classes):
            self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.counter = 0

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            self.counter += 1
            self.buffer.add_((probs - self.buffer) / self.counter)

        def output(self):
            probs, cls = self.buffer.max(1)
            return probs, cls

    class _VotingFusion:
        def __init__(self, x, classes):
            self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            probs, cls = probs.max(1, keepdim=True)

            self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
            self.probs.scatter_add_(1, cls, probs)

        def output(self):
            cls, idx = self.votes.max(1, keepdim=True)
            probs = self.probs / self.votes.clamp(min=1)
            probs = probs.gather(1, idx)
            return probs.squeeze(1), cls.squeeze(1)

    class _MaxFusion:
        def __init__(self, x, _):
            self.buffer_cls = x.new_zeros(x.size(0), x.size(2), x.size(3), dtype=torch.long)
            self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            max_prob, max_cls = probs.max(1)

            replace_idx = max_prob > self.buffer_prob
            self.buffer_cls[replace_idx] = max_cls[replace_idx]
            self.buffer_prob[replace_idx] = max_prob[replace_idx]

        def output(self):
            return self.buffer_prob, self.buffer_cls

    def __init__(self, body, head, head_channels, classes, fusion_mode="mean"):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.cls = nn.Conv2d(head_channels, classes, 1)

        self.classes = classes
        if fusion_mode == "mean":
            self.fusion_cls = SegmentationModule._MeanFusion
        elif fusion_mode == "voting":
            self.fusion_cls = SegmentationModule._VotingFusion
        elif fusion_mode == "max":
            self.fusion_cls = SegmentationModule._MaxFusion

    def _network(self, x, scale):
        if scale != 1:
            scaled_size = [s * scale for s in x.shape[-2:]]
<<<<<<< HEAD
            x_up = functional.upsample(x, size=scaled_size, mode='bilinear', align_corners=True)
=======
            x_up = functional.interpolate(x, size=scaled_size, mode='bilinear', align_corners=True)
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
        else:
            x_up = x

        x_up = self.body(x_up)
        x_up = self.head(x_up)
        sem_logits = self.cls(x_up)

        return x_up, sem_logits

    def forward(self, x, scales, do_flip=True):
        out_size = x.shape[-2:]
        fusion = self.fusion_cls(x, self.classes)
        x_up_all = []

        for scale in scales:
            # Main orientation
            x_up, sem_logits = self._network(x, scale)
<<<<<<< HEAD
            sem_logits = functional.upsample(sem_logits, size=out_size, mode='bilinear', align_corners=True)
            x_up_all.append(functional.upsample(x_up, size=out_size, mode='bilinear', align_corners=True).to("cpu"))
=======
            sem_logits = functional.interpolate(sem_logits, size=out_size, mode='bilinear', align_corners=True)
            x_up_all.append(functional.interpolate(x_up, size=out_size, mode='bilinear', align_corners=True).to("cpu"))
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
            fusion.update(sem_logits)
            

            # Flipped orientation
            if do_flip:
                # Main orientation
                x_up, sem_logits = self._network(flip(x, -1), scale)
<<<<<<< HEAD
                sem_logits = functional.upsample(sem_logits, size=out_size, mode='bilinear', align_corners=True)
                fusion.update(flip(sem_logits, -1))
                x_up_all.append(flip(functional.upsample(x_up, size=out_size, mode='bilinear', align_corners=True).to("cpu"), -1))
=======
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode='bilinear', align_corners=True)
                fusion.update(flip(sem_logits, -1))
                x_up_all.append(flip(functional.interpolate(x_up, size=out_size, mode='bilinear', align_corners=True).to("cpu"), -1))
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
        x_up = torch.cat(x_up_all, 1)
        return x_up, fusion.output()


def main():
    #object_class = [20,21,52,53,54,55,56,57,58,59,60,61,62,64]

    # Load configuration
    args = parser.parse_args()

    # Torch stuff
    torch.cuda.set_device(args.rank)
    cudnn.benchmark = True

    # Create model by loading a snapshot
    body, head, cls_state = load_snapshot(args.snapshot)
    model = SegmentationModule(body, head, 256, 65, args.fusion_mode)
    model.cls.load_state_dict(cls_state)
    model = model.cuda().eval()

    # Create data loader
    transformation = SegmentationTransform(
        1920,
        (0.41738699, 0.45732192, 0.46886091),
        (0.25685097, 0.26509955, 0.29067996),
    )
    for dir_ in os.listdir('/data/JAAD_clip_images'):
        dir_ = dir_.split('.')[0]
<<<<<<< HEAD
        os.system('mkdir -p /workspace/caozhangjie/inplace_abn/JAAD_features_occl/'+dir_)
        os.system('mkdir -p /workspace/caozhangjie/inplace_abn/JAAD_pos_occl/'+dir_)
        os.system('mkdir -p /workspace/caozhangjie/inplace_abn/JAAD_seg_occl/'+dir_)

    dataset = SegmentationDataset('/data/JAAD_clip_images', '/workspace/caozhangjie/inplace_abn/JAAD_vbb/vbb_full', '/workspace/caozhangjie/inplace_abn/JAAD_seg_occl', '/workspace/caozhangjie/inplace_abn/JAAD_features_occl', '/workspace/caozhangjie/inplace_abn/JAAD_pos_occl', transformation, "train", [args.start,args.start+60], 100)
=======
        os.system('mkdir -p /workspace/caozhangjie/inplace_abn/JAAD_features/'+dir_)
        os.system('mkdir -p /workspace/caozhangjie/inplace_abn/JAAD_pos/'+dir_)
        os.system('mkdir -p /workspace/caozhangjie/inplace_abn/JAAD_seg/'+dir_)

    dataset = SegmentationDataset('/data/JAAD_clip_images', '/workspace/caozhangjie/inplace_abn/JAAD_vbb/vbb_full', '/workspace/caozhangjie/inplace_abn/JAAD_seg', '/workspace/caozhangjie/inplace_abn/JAAD_features', '/workspace/caozhangjie/inplace_abn/JAAD_pos', transformation, "train", [args.start,args.start+30])
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        collate_fn=segmentation_collate,
        shuffle=False
    )

    # Run testing
    scales = [int(val) for val in args.scales.split(",")]
    with torch.no_grad():
        for batch_i, rec in enumerate(data_loader):
            print("Testing batch [{:3d}/{:3d}]".format(batch_i + 1, len(data_loader)))

            img = rec["img"].cuda(non_blocking=True)
            x_up, (probs, preds) = model(img, scales, args.flip)
<<<<<<< HEAD
            x_up = x_up.cpu()
            
=======
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f

            for i, (prob, pred) in enumerate(zip(torch.unbind(probs, dim=0), torch.unbind(preds, dim=0))):
                out_size = rec["size"][i]
                bboxes = rec["pos"][i]
                ped_appear = rec['ped_appear'][i]
                #img_name = rec["meta"][i]["idx"]

                # Save prediction
                prob = prob.cpu()
                pred = pred.cpu()
                pred_img = get_pred_image(pred, out_size, args.output_mode == "palette")
                pred_img.save(rec["seg_out"][i])
<<<<<<< HEAD
                #torch.save(x_up.cpu(), rec['raw_feature_out'][i])
                x_up = x_up.cpu()
=======
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
                width_dict, height_dict = get_box(pred)
                features = []
                spp_scales = [4,2,1]
                height_threshold = 15
                width_threshold = 15
                feature_dim = x_up.size(1) * sum([scale * scale for scale in spp_scales])
                position = []
                for j in range(65):
                    if j != 19:
                        if width_dict[j][0] != -1:
                            if not ((width_dict[j][1]+1 - width_dict[j][0] < width_threshold) or (height_dict[j][1]+1 - height_dict[j][0] < height_threshold)):
                                raw_feature = x_up[:,:,height_dict[j][0]:height_dict[j][1]+1, width_dict[j][0]:width_dict[j][1]+1]
                                position.append([width_dict[j][0], height_dict[j][0], \
                                width_dict[j][1]+1-width_dict[j][0], height_dict[j][1]+1-height_dict[j][0]])
<<<<<<< HEAD
                                spp_feature = spatial_pyramid_pool(raw_feature.cuda().detach(), 1, [raw_feature.size(2), raw_feature.size(3)], [4, 2, 1]).cpu()
=======
                                spp_feature = spatial_pyramid_pool(raw_feature, 1, [raw_feature.size(2), raw_feature.size(3)], [4, 2, 1])
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
                                features.append(spp_feature)
                            else:
                                features.append(torch.zeros([1, feature_dim]))
                                position.append([-1, -1, -1, -1])
                        else:
                            features.append(torch.zeros([1, feature_dim]))
                            position.append([-1, -1, -1, -1])
                for j in range(ped_appear.shape[0]):
                    bbox = bboxes[j, :]
                    if ped_appear[j] > 0 and (bbox[2]>=width_threshold and bbox[3] >=height_threshold):
                        ped_raw_feature = x_up[:,:,bbox[1]:bbox[1]+bbox[3], \
                                            bbox[0]:bbox[0]+bbox[2]]
<<<<<<< HEAD
                        features.append(spatial_pyramid_pool(ped_raw_feature.cuda().detach(), \
                            1, [ped_raw_feature.size(2), ped_raw_feature.size(3)], [4, 2, 1]).cpu())
=======
                        features.append(spatial_pyramid_pool(ped_raw_feature, \
                            1, [ped_raw_feature.size(2), ped_raw_feature.size(3)], [4, 2, 1]))
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
                    else:
                        features.append(torch.zeros([1, feature_dim]))
                    position.append([bbox[i] for i in range(4)])
                features = torch.cat(features, dim=0)
                position = torch.from_numpy(np.array(position))
                #print(features.size())
                #print(position.size())
                torch.save(features, rec['feature_out'][i])
                torch.save(position, rec['pos_out'][i])
<<<<<<< HEAD
                
=======
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f

            #    # Optionally save probabilities
            #    if args.output_mode == "prob":
            #        prob_img = get_prob_image(prob, out_size)
            #        prob_img.save(path.join(args.output, img_name + "_prob.png"))

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)//2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)//2
<<<<<<< HEAD
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad)).cuda()
=======
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
>>>>>>> a85257dde9e4ae6445a0d39a6bdfe48254778e0f
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
        else:
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

def get_box(pred):
    width = pred.size(1)
    height = pred.size(0)
    width_dict = {i:[-1, -1] for i in range(65)}
    height_dict = {i:[-1, -1] for i in range(65)}
    for i in range(height):
        for j in range(width):
            if height_dict[int(pred[i,j])][0] == -1:
                height_dict[int(pred[i,j])][0] = i
    for j in range(width):
        for i in range(height):
            if width_dict[int(pred[i,j])][0] == -1:
                width_dict[int(pred[i,j])][0] = j
    for i in range(height-1, -1, -1):
        for j in range(width):
            if height_dict[int(pred[i,j])][1] == -1:
                height_dict[int(pred[i,j])][1] = i
    for j in range(width-1, -1, -1):
        for i in range(height):
            if width_dict[int(pred[i,j])][1] == -1:
                width_dict[int(pred[i,j])][1] = j
    return width_dict, height_dict
             
        

def load_snapshot(snapshot_file):
    """Load a training snapshot"""
    print("--- Loading model from snapshot")

    # Create network
    norm_act = partial(ABN, activation="leaky_relu", slope=.01)
    body = models.__dict__["net_wider_resnet38_a2"](norm_act=norm_act, dilation=(1, 2, 4, 4))
    head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))

    # Load snapshot and recover network state
    data = torch.load(snapshot_file)
    body.load_state_dict(data["state_dict"]["body"])
    head.load_state_dict(data["state_dict"]["head"])

    return body, head, data["state_dict"]["cls"]


_PALETTE = np.array([[165, 42, 42],
                     [0, 192, 0],
                     [196, 196, 196],
                     [190, 153, 153],
                     [180, 165, 180],
                     [90, 120, 150],
                     [102, 102, 156],
                     [128, 64, 255],
                     [140, 140, 200],
                     [170, 170, 170],
                     [250, 170, 160],
                     [96, 96, 96],
                     [230, 150, 140],
                     [128, 64, 128],
                     [110, 110, 110],
                     [244, 35, 232],
                     [150, 100, 100],
                     [70, 70, 70],
                     [150, 120, 90],
                     [220, 20, 60],
                     [255, 0, 0],
                     [255, 0, 100],
                     [255, 0, 200],
                     [200, 128, 128],
                     [255, 255, 255],
                     [64, 170, 64],
                     [230, 160, 50],
                     [70, 130, 180],
                     [190, 255, 255],
                     [152, 251, 152],
                     [107, 142, 35],
                     [0, 170, 30],
                     [255, 255, 128],
                     [250, 0, 30],
                     [100, 140, 180],
                     [220, 220, 220],
                     [220, 128, 128],
                     [222, 40, 40],
                     [100, 170, 30],
                     [40, 40, 40],
                     [33, 33, 33],
                     [100, 128, 160],
                     [142, 0, 0],
                     [70, 100, 150],
                     [210, 170, 100],
                     [153, 153, 153],
                     [128, 128, 128],
                     [0, 0, 80],
                     [250, 170, 30],
                     [192, 192, 192],
                     [220, 220, 0],
                     [140, 140, 20],
                     [119, 11, 32],
                     [150, 0, 255],
                     [0, 60, 100],
                     [0, 0, 142],
                     [0, 0, 90],
                     [0, 0, 230],
                     [0, 80, 100],
                     [128, 64, 64],
                     [0, 0, 110],
                     [0, 0, 70],
                     [0, 0, 192],
                     [32, 32, 32],
                     [120, 10, 10]], dtype=np.uint8)
_PALETTE = np.concatenate([_PALETTE, np.zeros((256 - _PALETTE.shape[0], 3), dtype=np.uint8)], axis=0)
_PALETTE = ImagePalette.ImagePalette(
    palette=list(_PALETTE[:, 0]) + list(_PALETTE[:, 1]) + list(_PALETTE[:, 2]), mode="RGB")


def get_pred_image(tensor, out_size, with_palette):
    tensor = tensor.numpy()
    if with_palette:
        img = Image.fromarray(tensor.astype(np.uint8), mode="P")
        img.putpalette(_PALETTE)
    else:
        img = Image.fromarray(tensor.astype(np.uint8), mode="L")

    return img.resize(out_size, Image.NEAREST)


def get_prob_image(tensor, out_size):
    tensor = (tensor * 255).to(torch.uint8)
    img = Image.fromarray(tensor.numpy(), mode="L")
    return img.resize(out_size, Image.NEAREST)


if __name__ == "__main__":
    main()
