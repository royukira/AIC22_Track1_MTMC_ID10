from email import header

from scipy.fftpack import dst


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_train_gt_bbox.py
@Time    :   2022/03/12 15:19:16
@Author  :   Roy Cheung 
@Version :   0.1
@Contact :   zhang.rui_sh@tslsmart.com
@License :   Copyright (c) Terminus AI. All rights reserved.
'''
# here put the import lib
import os
import cv2
import numpy as np
from tqdm import tqdm

def load_gt(gt_path) -> dict:
    gt_dict = {}    # key is int(fid), value is dict(oid:gt_info)
    with open(gt_path, 'r') as f_r:
        for line in f_r.readlines():
            gt = line.split("\n")[0]
            gt_info = gt.split(",")

            fid = int(gt_info[0])
            oid = int(gt_info[1])
            if fid in gt_dict:
                if oid in gt_dict[fid]:
                    raise AssertionError("Object ID is duplicate in the same frame!")
                gt_dict[fid].setdefault(
                    oid, gt_info[2:]
                )
            else:
                gt_dict.setdefault(
                    fid, 
                    {oid: gt_info[2:]}
                )
    
    return gt_dict

def crop_obj(img_path, bbox):
    l, t, w, h = bbox
    img = cv2.imread(img_path)
    
    if img is not None:
        return img[t:t+h, l:l+w]
    else:
        raise AssertionError(f"Image is none. Please check the image path {img_path} if it's vaild.")

def save_img(img, frame_id, obj_id, dst_dir, postfix='png'):
    if os.path.exists(dst_dir) is False:
            os.makedirs(dst_dir)
    dst_path = os.path.join(dst_dir, f"{frame_id}_{obj_id}.{postfix}")
    cv2.imwrite(dst_path, img)

def run(img_dir, gt_dir, scene_list, dst_path, postfix='png'):
    for sid in range(len(scene_list)):
        sname = scene_list[sid]     # S0X
        simg_dir = os.path.join(img_dir, sname)
        sgt_dir = os.path.join(gt_dir, sname)
        sdst_dir = os.path.join(dst_path, sname)

        # 检查cam是否一致
        if os.listdir(simg_dir) != os.listdir(sgt_dir):
            raise AssertionError(f"The num of camera in {sname} is conflicting.")

        if os.path.exists(sdst_dir) is False:
            os.makedirs(sdst_dir)
        
        cam_list = os.listdir(simg_dir)
        with tqdm(total=len(cam_list)) as _tqdm:
            _tqdm.set_description('Scene: {}/{}'.format(sid + 1, len(scene_list)))
            for cid in range(len(cam_list)):
                cname = cam_list[cid]
                cimg_dir = os.path.join(simg_dir, f"{cname}/img1")
                cgt_path = os.path.join(sgt_dir, f"{cname}/gt/gt.txt")
                cdst_dir = os.path.join(sdst_dir, cname)

                if os.path.exists(cdst_dir) is False:
                    os.makedirs(cdst_dir)

                img_names = os.listdir(cimg_dir)
                gt_info = load_gt(cgt_path)
                
                _tqdm.set_postfix(num_of_img='{}'.format(len(img_names)))
                for img_idx in range(len(img_names)):
                    iname = img_names[img_idx]
                    ipath = os.path.join(cimg_dir, iname)
                    frame_id = int(iname.split(".jpg")[0].split("img")[-1]) + 1     # e.g img00001.jpg -> 2, the gt frame id starts with 1
                    
                    if frame_id not in gt_info:
                        continue

                    # convert to int
                    for gt_obj_id, gt in gt_info[frame_id].items():
                        gt_obj_bbox = list(map(int, gt[0:4]))        # ltwh
                        crop_img = crop_obj(ipath, gt_obj_bbox)
                        save_img(crop_img, frame_id, gt_obj_id, cdst_dir, postfix)

                _tqdm.update(1)


if __name__ == "__main__":
    dst_path = "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/validation/det_feat/det"
    train_data_path = "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/validation/"
    img_path = os.path.join(train_data_path, 'images/validation')
    if os.path.exists(img_path) is False:
        raise AssertionError(f"{img_path} does not exist.")

    scene_list = [s for s in os.listdir(img_path) if len(s.split("S")) == 2]    # 目前S0X都是两位数

    run(img_path, train_data_path, scene_list, dst_path)
    
