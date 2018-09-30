import os
import pickle
import numpy as np
from multiprocessing import Pool

def extract_bbox(params):
        dir_ = params["dir"]
        video = params['video']
        data = pickle.load(open("/data/JAAD_result/"+dir_+"/"+video, "rb"))
        index_ = data['cls_boxes'][1][:,4] >= 0.9
        #index_ = [index_[i] for i in range(index_.shape[0]) if index_[i] == True]
        #print(data.shape)
        need_boxes = data['cls_boxes'][1][index_, :]
        np.save("./JAAD_pred_box/"+dir_+"/"+video.split(".")[0]+".npy", need_boxes)

params_list = []
for dir_ in os.listdir("/data/JAAD_result/"):
    os.system("mkdir -p "+"./JAAD_pred_box/"+dir_)
    for video in os.listdir("/data/JAAD_result/"+dir_):
        params_list.append({"dir":dir_, "video":video})
p = Pool(8)
p.map(extract_bbox, params_list)

