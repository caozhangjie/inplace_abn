import numpy as np
from multiprocessing import Pool
import os

pedestrian_action = {'standing': 0, 'walking': 1, 'speed up': 2, 'nod': 3, 'looking': 4, 'handwave': 5, 'clear path': 6, 'crossing': 7, 'slow down': 8}
label_cross = [2,6,7,8]

def transfer_label(params):
    f_name = params['in_name']
    out_name = params['out_name']
    label = np.load(f_name)
    c_label = np.zeros([label.shape[0]])
    for i in range(label.shape[0]):
        for j in label_cross:
            if label[i][j] > 0:
                c_label[i] = 1
    np.save(out_name, c_label)

params_list = []
os.system('mkdir -p /data/JAAD_cross_label')
for dir_ in os.listdir('/data/JAAD_behavioral_encode'):
    os.system('mkdir -p /data/JAAD_cross_label/'+dir_)
    for b_name in os.listdir('/data/JAAD_behavioral_encode/'+dir_):
        if 'pedestrian' in b_name:
            params_list.append({'in_name':'/data/JAAD_behavioral_encode/'+dir_+'/'+b_name, 'out_name':'/data/JAAD_cross_label/'+dir_+'/'+b_name})
p = Pool(8)
p.map(transfer_label, params_list)
