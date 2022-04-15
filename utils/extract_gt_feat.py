#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   extract_gt_feat.py
@Time    :   2022/03/14 16:22:25
@Author  :   Roy Cheung 
@Version :   0.1
@Contact :   zhang.rui_sh@tslsmart.com
@License :   Copyright (c) Terminus AI. All rights reserved.
'''
# here put the import lib

"""Extract image feature for both det/mot image feature."""

import os
import pickle
import time
from glob import glob
from itertools import cycle
from multiprocessing import Pool, Queue
import tqdm

import torch
from PIL import Image
import torchvision.transforms as T
import sys
sys.path.append('../')
from reid.reid_inference.reid_model import build_reid_model
from config import cfg

BATCH_SIZE = 64
NUM_PROCESS = 8
def chunks(l):
    return [l[i:i+BATCH_SIZE] for i in range(0, len(l), BATCH_SIZE)]

class ReidFeature():
    """Extract reid feature."""

    def __init__(self, gpu_id, _mcmt_cfg):
        print("init reid model")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.model, self.reid_cfg = build_reid_model(_mcmt_cfg)
        device = torch.device('cuda')
        self.model = self.model.to(device)
        self.model.eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_transforms = T.Compose([T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),\
                              T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def extract(self, img_path_list):
        """Extract image feature with given image path.
        Feature shape (2048,) float32."""

        img_batch = []
        for img_path in img_path_list:
            img = Image.open(img_path).convert('RGB')
            img = self.val_transforms(img)
            img = img.unsqueeze(0)
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0)

        with torch.no_grad():
            img = img.to('cuda')
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == 'yes': flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy()
        return feat


def init_worker(gpu_id, _cfg):
    """init worker."""

    # pylint: disable=global-variable-undefined
    global model
    model = ReidFeature(gpu_id.get(), _cfg)


def process_input_by_worker_process(image_path_list):
    """Process_input_by_worker_process."""

    reid_feat_numpy = model.extract(image_path_list)
    feat_dict = {}
    for index, image_path in enumerate(image_path_list):
        feat_dict[image_path] = reid_feat_numpy[index]
    return feat_dict


def save_feature(cfg, pool_output, scene_id, postfix='png'):
    """Save feature."""
    output_dir = os.path.join(cfg.OUTPUT_DIR, scene_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    all_feat_dic = {}
    for sample_dic in pool_output:
        for image_path, feat in sample_dic.items():
            relative_path = image_path.split(cfg.DET_IMG_DIR)[-1]
            image_name = os.path.basename(image_path)                       # e.g. fid_oid.png
            
            cam_id = relative_path.split("/")[1]
            if cam_id[0] != "c":
                raise AttributeError(f"Cam ID format(c00x) is wrong! {cam_id}")

            fid, oid = image_name.split(f".{postfix}")[0].split("_")        # frame id and obj id
            
            all_feat_dic[f"{cam_id}_{fid}_{oid}"] = {
                'img_rpath': relative_path,                                 # relative path
                'scene': scene_id,
                'cam': cam_id,
                'frame': int(fid),
                'ID': int(oid),
                'feature': feat
            }
    
    dst_path = os.path.join(output_dir, f'{scene_id}_all_feat.pkl')
    pickle.dump(all_feat_dic, open(dst_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print('save pickle in %s' % dst_path)


def load_all_data(data_path, postfix='png'):
    """Load all mode data."""

    image_list = []
    for cam in os.listdir(data_path):
        image_dir = os.path.join(data_path, cam)
        cam_image_list = glob(image_dir+f'/*.{postfix}')
        cam_image_list = sorted(cam_image_list)
        print(f'{len(cam_image_list)} images for {cam}')
        image_list += cam_image_list
    print(f'{len(image_list)} images in total')
    return image_list

def extract_image_feat(_cfg, postfix='png'):
    scene_list = [s for s in os.listdir(_cfg.DET_IMG_DIR) if len(s.split("S")) == 2]    # 目前S0X都是两位数
    for sid in range(len(scene_list)):
        sname = scene_list[sid]     # S0X
        simg_dir = os.path.join(_cfg.DET_IMG_DIR, sname) 
        image_list = load_all_data(simg_dir, postfix)
        image_list = sorted(image_list)
        chunk_list = chunks(image_list)
        
        num_process = NUM_PROCESS
        gpu_ids = Queue()
        gpu_id_cycle_iterator = cycle(range(0, 8))
        for _ in range(num_process):
            gpu_ids.put(next(gpu_id_cycle_iterator))

        process_pool = Pool(processes=num_process, initializer=init_worker, initargs=(gpu_ids, _cfg, ))
        start_time = time.time()
        pool_output = list(tqdm.tqdm(process_pool.imap_unordered(\
                                    process_input_by_worker_process, chunk_list),
                                    total=len(chunk_list)))
        process_pool.close()
        process_pool.join()

        print('%.4f s' % (time.time() - start_time))
        
        # debug
        # for image_path in chunk_list[0]:
        #     relative_path = image_path.split(cfg.DET_IMG_DIR)[-1]
        #     image_name = os.path.basename(image_path)                       # e.g. fid_oid.png
        #     cam_id = relative_path.split("/")[1]
        #     print(f"{cam_id}_{image_name}")
        #     if cam_id[0] != "c":
        #         raise AttributeError(f"Cam ID format(c00x) is wrong! {cam_id}")
        

        save_feature(_cfg, pool_output, scene_id=sname)
            


def main():
    """Main method."""
    cfg.merge_from_file(f'{sys.argv[1]}')
    #cfg.merge_from_file('/home/zhangrui/AIC21-MTMC/config/train_data/reid1.yml')
    cfg.freeze()
    extract_image_feat(cfg)


if __name__ == "__main__":
    main()
