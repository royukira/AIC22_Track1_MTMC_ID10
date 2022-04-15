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

def load_pred(pred_result_path) -> dict:
    pred_dict = {}    # key is (cid, fid), value is dict(oid:gt_info)
    with open(pred_result_path, 'r') as f_r:
        for line in f_r.readlines():
            pred = line.split("\n")[0]
            pred_info = pred.split(" ")

            cid = int(pred_info[0])
            oid = int(pred_info[1])
            fid = int(pred_info[2])

            if (cid,fid) in pred_dict:
                if oid in pred_dict[(cid,fid)]:
                    # raise AssertionError("Object ID is duplicate in the same frame!")
                    continue
                pred_dict[(cid,fid)].setdefault(
                    oid, pred_info[3:]
                )
            else:
                pred_dict.setdefault(
                    (cid,fid), 
                    {oid: pred_info[3:]}
                )
    
    return pred_dict

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

def run(img_dir, pred_path, dst_path, postfix='png'):
    sname = 'S06'
    simg_dir = os.path.join(img_dir, sname)
    sdst_dir = os.path.join(dst_path, sname)
    
    if os.path.exists(sdst_dir) is False:
            os.makedirs(sdst_dir)

    pred_info = load_pred(pred_path)

    cam_list = os.listdir(simg_dir)
    with tqdm(total=len(cam_list)) as _tqdm:
        for cid in range(len(cam_list)):
            _tqdm.set_description('Cam: {}/{}'.format(cid + 1, len(cam_list)))
            cname = cam_list[cid]
            cnum = int(cname.split("c")[-1])
            cimg_dir = os.path.join(simg_dir, f"{cname}/img1")
            cdst_dir = os.path.join(sdst_dir, cname)
            if os.path.exists(cdst_dir) is False:
                os.makedirs(cdst_dir)
            img_names = os.listdir(cimg_dir)
            for img_idx in range(len(img_names)):
                iname = img_names[img_idx]
                ipath = os.path.join(cimg_dir, iname)
                frame_id = int(iname.split(".jpg")[0].split("img")[-1]) + 1     # e.g img00001.jpg -> 2, the gt frame id starts with 1

                if (cnum, frame_id) not in pred_info:
                    continue

                # convert to int
                for pred_obj_id, pred in pred_info[(cnum, frame_id)].items():
                    pred_obj_bbox = list(map(int, pred[0:4]))        # ltwh
                    crop_img = crop_obj(ipath, pred_obj_bbox)
                    save_img(crop_img, frame_id, pred_obj_id, cdst_dir, postfix)

if __name__ == "__main__":
    dst_path = "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/test_baseline/train_data/det"
    train_data_path = "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/test/S06"
    pred_path = "/home/zhangrui/AIC21-MTMC/results/base_bytetrack/0.4_0.8_iou_occ_cleannosplit_track1.txt"
    img_path = os.path.join(train_data_path, 'images/test')
    if os.path.exists(img_path) is False:
        raise AssertionError(f"{img_path} does not exist.")

    run(img_path, pred_path, dst_path)
    
