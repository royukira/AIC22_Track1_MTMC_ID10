#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   merge_gt_feat.py
@Time    :   2022/03/14 16:22:35
@Author  :   Roy Cheung 
@Version :   0.1
@Contact :   zhang.rui_sh@tslsmart.com
@License :   Copyright (c) Terminus AI. All rights reserved.
'''
# here put the import lib


import os
import pickle
import sys
import json
from sklearn import preprocessing
import numpy as np
sys.path.append('../')

def check_vaild(sfeat_map, ensemble_num):
    for k, v in sfeat_map.items():
        if len(v) != ensemble_num:
            raise AssertionError("Missing some features.")

# 转成单镜头轨迹序列（帧数从小到大）形式存储
def cvt_to_sequence(merge_feat_dict:dict) -> dict:
    new_feat_dict = {}
    for _, info in merge_feat_dict.items():
        cam_id = info['cam']
        obj_id = info['ID']
        frame_id = int(info['frame'])
        new_key = f"{cam_id}_{obj_id}"
        if new_key not in new_feat_dict:
            new_feat_dict.setdefault(
                new_key, {frame_id: info}
            )
        else:
            new_feat_dict[new_key].setdefault(
                frame_id, info
            )

    for seq_key, seq_info in new_feat_dict.items():
        new_seq_info_list = [seq_info[k] for k in sorted(seq_info.keys())]
        new_feat_dict[seq_key] = new_seq_info_list
    
    return new_feat_dict


def merge_feat(feat_dir, dst_dir, ensemble_seq = ['reid1', 'reid2', 'reid3'], merge_to_seq=True):
    """Save feature."""
    # NOTE: modify the ensemble list here
    sfeat_dir_map = {}  # key: S0x, val: path of scene dir
    for e in ensemble_seq:
        fdir = os.path.join(feat_dir, e)
        scene_list = os.listdir(fdir)
        for s in scene_list:
            if s in sfeat_dir_map:
                sfeat_dir_map[s].append(os.path.join(fdir, s))
            else:
                sfeat_dir_map.setdefault(s, [os.path.join(fdir, s)])
        
    print(json.dumps(sfeat_dir_map, indent=4, ensure_ascii=False))

    # 检查是否有一样的特征数量
    check_vaild(sfeat_dir_map, len(ensemble_seq))

    save_dir = os.path.join(dst_dir, "merge_feat")
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    print("merging...")
    for k, v in sfeat_dir_map.items():
        merge_feat_dict = {}
        feat_path_list = [os.path.join(p, f'{k}_all_feat.pkl') for p in v]
        fdata_list = []
        for fname in feat_path_list:
            fdata = pickle.load(open(fname, 'rb'))
            fdata_list.append(fdata)
        
        for fkey in fdata_list[0].keys():
            if fkey not in merge_feat_dict:
                merge_feat_dict.setdefault(
                    fkey, fdata_list[0][fkey]
                )

            feat_list =[]
            for fd in fdata_list:
                feat_list.append(fd[fkey]['feature'])

            if len(feat_list) != len(ensemble_seq):
                raise AssertionError("Missing some features")
            
            feat_array = np.array(feat_list)
            feat_array = preprocessing.normalize(feat_array, norm='l2', axis=1)
            feat_array = np.mean(feat_array, axis=0)
            merge_feat_dict[fkey]['feature'] = feat_array

        save_path = os.path.join(save_dir, f"{k}_merge_feat.pkl")
        
        # 转成单镜头轨迹序列（帧数从小到大）形式存储
        if merge_to_seq:
            merge_feat_dict = cvt_to_sequence(merge_feat_dict)
        
        pickle.dump(merge_feat_dict, open(save_path, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % save_path)

if __name__ == "__main__":
    src_dir = "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/test_baseline/train_data/feat/"
    dst_dir = "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/test_baseline/train_data/feat/"
    merge_feat(src_dir, dst_dir)