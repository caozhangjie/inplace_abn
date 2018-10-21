import numpy as np
import os

def gen_intent_label(in_name, out_dir, num_frames):
    appear_labels = np.loadtxt(in_name)
    if appear_labels.shape[0] == 0:
        return
    elif len(appear_labels.shape) == 1:
        appear_labels = appear_labels.reshape([1, appear_labels.shape[0]])
    intent_labels = np.zeros([appear_labels.shape[0], num_frames]).astype(np.uint8)
    for i in range(appear_labels.shape[0]):
        if appear_labels[i, 2] > 0:
            intent_labels[i, int(appear_labels[i,0]):int(appear_labels[i,3])] = 1
    for i in range(num_frames):
        np.save(out_dir+"/{:03d}.npy".format(i), intent_labels[:,i])

for dir_ in os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data'):
    gen_intent_label('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/appear_cross_list.txt', '/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/intent_labels', len(os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/pos')))
