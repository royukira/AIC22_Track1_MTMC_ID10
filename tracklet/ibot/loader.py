# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import math
import pickle
import numpy as np

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple

SCENE_CAM = {
    'S01': ['c001','c002','c003','c004','c005'],
    'S02': ['c006','c007','c008','c009'],
    'S03': ['c010','c011','c012','c013','c014','c015'],
    'S04': ['c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023',
            'c024','c025','c026','c027','c028','c029','c030','c031','c032',
            'c033','c034','c035','c036','c037','c038','c039','c040'],
    'S05': ['c010','c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023',
            'c024','c025','c026','c027','c028','c029','c033','c034','c035','c036'],
}


class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)


# ===========================================================================================
#                                   For CityFlow2                                          ==
# ===========================================================================================

class TrackletEmbeddingInstance(Dataset):
    def __init__(self, root_dir, seq_len, return_index=True):
         self.root_dir = root_dir
         self.seq_len = seq_len
         self.all_tracklet_embed = []    
         
         self.min_id = 1e6
         self.max_id = -1
         self.epoch = 0
         self.return_index = return_index

         print("Loading tracklet features...")
         self.load_tracklet_feats()
         print(f"Total: {self.__len__()}")
    
    def resize_seq_len(self, embedding_feat:np.ndarray):
        if embedding_feat.shape[0] == self.seq_len:
            return embedding_feat

        # 过长裁剪头尾段
        if embedding_feat.shape[0] > self.seq_len:
            start_offset = self.seq_len // 2
            end_offset = embedding_feat.shape[0] - self.seq_len // 2
            if self.seq_len % 2 != 0:
                new_seq_feat = np.vstack(
                    (embedding_feat[:start_offset+1,:], 
                    embedding_feat[end_offset:,:])
                )
            else:
                new_seq_feat = np.vstack(
                    (embedding_feat[:start_offset,:], 
                    embedding_feat[end_offset:,:])
                )
            return new_seq_feat

        # Padding 0
        if embedding_feat.shape[0] < self.seq_len:
            pad_zero = np.zeros((self.seq_len-embedding_feat.shape[0], embedding_feat.shape[1]))
            new_seq_feat = np.vstack((embedding_feat, pad_zero))
            return new_seq_feat

    def _data_load(self, root_dirs:list):
        for dir_path in root_dirs:
            scene_pkls = os.listdir(dir_path)
            for sp in scene_pkls:
                if sp.split(".")[-1] != 'pkl':
                    continue
                data = pickle.load(open(os.path.join(dir_path, sp), 'rb'))
                for k, v in data.items():
                    sname = v['seq_info'][0]['scene']   # Scene name, e.g. S0X
                    cname = v['seq_info'][0]['cam']     # Cam name, e.g. c00x
                    oid = v['seq_info'][0]['ID']        # Object name, e.g. x  dtype:int
                    new_data_dict = {
                        'name': k,
                        'sid': sname,
                        'cid': cname,
                        'id': oid,
                        'embedding': self.resize_seq_len(v['seq_feat'])
                    }

                    if oid < self.min_id:
                        self.min_id = oid
                    if oid > self.max_id:
                        self.max_id = oid
                    self.all_tracklet_embed.append(new_data_dict)

    def load_tracklet_feats(self):
        root_dirs = self.root_dir if isinstance(self.root_dir, list) else [self.root_dir]
        self._data_load(root_dirs)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.all_tracklet_embed)

    def __getitem__(self, index):
        data = self.all_tracklet_embed[index]
        data_id = data['id']
        data_embed = data['embedding']

        embeds = torch.from_numpy(data_embed.copy())
        if self.return_index:
            return embeds, data_id, index
        return embeds, data_id


class TrackletEmbeddingMask(Dataset):
    def __init__(self, root_dir, seq_len, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_start_epoch=0, transform_type:str="reverse", embed_type="standard"):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_start_epoch = pred_start_epoch

        self.min_id = 1e6
        self.max_id = -1

        #self.embed_type = embed_type
        self.transform_type = transform_type

        self.all_scene_seq_feats = {}
        self.all_tracklet_embed = []    
        print("Loading tracklet features...")
        self.load_tracklet_feats()
        print(f"Total: {self.__len__()}")

    def resize_seq_len(self, embedding_feat:np.ndarray):
        if embedding_feat.shape[0] == self.seq_len:
            return embedding_feat
        
        # 裁剪头尾段
        if embedding_feat.shape[0] > self.seq_len:
            start_offset = self.seq_len // 2
            end_offset = embedding_feat.shape[0] - self.seq_len // 2
            if self.seq_len % 2 != 0:
                new_seq_feat = np.vstack(
                    (embedding_feat[:start_offset+1,:], 
                    embedding_feat[end_offset:,:])
                )
            else:
                new_seq_feat = np.vstack(
                    (embedding_feat[:start_offset,:], 
                    embedding_feat[end_offset:,:])
                )
            return new_seq_feat

        # Padding 0
        if embedding_feat.shape[0] < self.seq_len:
            pad_zero = np.zeros((self.seq_len-embedding_feat.shape[0], embedding_feat.shape[1]))
            new_seq_feat = np.vstack((embedding_feat, pad_zero))
            return new_seq_feat
    

    def concat_seq_embed(self, embedding_feats:Tuple[np.ndarray]):
        total_len = 0
        for e in embedding_feats:
            total_len += e.shape[0]
        
        if total_len <= self.seq_len:
            return np.vstack(embedding_feats)

        patch_num = len(embedding_feats)
        each_patch_len = self.seq_len // patch_num
        
        len_cnt = 0
        new_embed_feat = []
        for pid in range(patch_num):
            embed_feat = embedding_feats[pid]
            embed_len = embed_feat.shape[0]
            if (pid + 1) == patch_num:
                # The last embedding
                remain_patch_len = self.seq_len - len_cnt
                if embed_len <= remain_patch_len:
                    new_embed_feat.append(embed_feat)
                else:
                    cut_start_offset = random.randint(0, embed_len - remain_patch_len)
                    new_embed_feat.append(embed_feat[cut_start_offset:cut_start_offset+remain_patch_len,:])
            else:
                if embed_len <= each_patch_len:
                    new_embed_feat.append(embed_feat)
                    len_cnt += embed_len
                else:
                    cut_start_offset = random.randint(0, embed_len - each_patch_len)
                    new_embed_feat.append(embed_feat[cut_start_offset:cut_start_offset+each_patch_len,:])
                    len_cnt += each_patch_len
        
        new_embed_feat = np.vstack(new_embed_feat)

        return new_embed_feat

    # 顺序掉转，相当于倒放tracklet（是否有更多数据增强方法）
    def aug_reverse_seq_embed(self, embedding_feat:np.ndarray, **kwargs):
        # print("reverse")
        return embedding_feat[::-1, :] 

    # 用同S同ID，不同CAM下作为增强样本，增加全局视觉关联性，提高CLS训练难度
    def aug_diff_cam_seq_embed(self, sid, cid, oid, select_k=1, **kwargs):
        cam_seq_feats = self.all_scene_seq_feats[sid]
        cam_list = SCENE_CAM[sid].copy()
        cam_list.remove(cid)
        random.shuffle(cam_list)

        select_cnt = 0
        select_embeds = []
        while select_cnt < select_k:
            temp_cam_list = cam_list.copy()
            isfind = False
            for cam in temp_cam_list:
                query_name = f"{sid}_{cam}_{oid}"
                if query_name in cam_seq_feats:
                    # print(f"diff cam:{sid}_{cid}_{oid} -> {query_name}")
                    select_embeds.append(cam_seq_feats[query_name])
                    cam_list.remove(cam)
                    select_cnt += 1
                    isfind = True
                    break
            
            # 有些ID只有两个cam的序列
            if isfind is False:
                temp_embeds = self.aug_reverse_seq_embed(select_embeds[-1])
                select_embeds.append(temp_embeds)
                select_cnt += 1
            
        assert select_k == len(select_embeds)
        if select_k == 1:
            return select_embeds[0]
        return select_embeds

    # same S, same C, random/same ID
    def aud_randid_load(self, random_prob=0.1):
        pass

    # same S, different C, random/same ID
    def aud_multicam_randid_load(self, random_prob=0.1):
        pass

    # 加干扰，对抗训练
    # TODO:
    def aug_adversarial(self):
        pass
    
    # same S, same C, same ID
    def _data_load(self, root_dirs:list):
        for dir_path in root_dirs:
            scene_pkls = os.listdir(dir_path)
            for sp in scene_pkls:
                if sp.split(".")[-1] != 'pkl':
                    continue
                scene_name = sp.split("_")[0]
                temp_cam_data = {}
                data = pickle.load(open(os.path.join(dir_path, sp), 'rb'))
                for _, v in data.items():
                    sname = v['seq_info'][0]['scene']   # Scene name, e.g. S0X
                    cname = v['seq_info'][0]['cam']     # Cam name, e.g. c00x
                    oid = v['seq_info'][0]['ID']        # Object name, e.g. x  dtype:int

                    frames = []
                    for si in v['seq_info']:
                        frames.append(int(si['frame']))
                    frames.sort()
                    
                    if sname != scene_name:
                        raise AssertionError("Scene mismatach.")

                    new_key_name = f"{sname}_{cname}_{oid}"
                    new_data_dict = {
                        'name': new_key_name,
                        'sid': sname,
                        'cid': cname,
                        'id': oid,
                        'frames': frames,
                        'embedding': v['seq_feat'],
                    }
                    self.all_tracklet_embed.append(new_data_dict)
                    
                    if new_key_name in temp_cam_data:
                        raise AssertionError(f"Duplicate key: {new_key_name}")
                    temp_cam_data.setdefault(new_key_name, v['seq_feat'])

                    if oid < self.min_id:
                        self.min_id = oid
                    if oid > self.max_id:
                        self.max_id = oid

                # TODO: high memory consumption
                self.all_scene_seq_feats.setdefault(scene_name, temp_cam_data)

    def load_tracklet_feats(self):
        root_dirs = self.root_dir if isinstance(self.root_dir, list) else [self.root_dir]
        self._data_load(root_dirs)


    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.all_tracklet_embed)

    def __getitem__(self, index):
        data = self.all_tracklet_embed[index]
        data_name = data['name']
        data_id = data['id']
        data_cid = data['cid']
        data_sid = data['sid']
        data_frames = data['frames']
        data_embed = data['embedding']

        # Transform
        if self.transform_type == "reverse":
            aug_embed = self.aug_reverse_seq_embed(data_embed)
        elif self.transform_type == "multicam":
            aug_embed = self.aug_diff_cam_seq_embed(data_sid, data_cid, data_id, select_k=1)
        elif self.transform_type == "concat_multicam":
            # Randomly choose the other cam
            other_embed_1, other_embed_2 = self.aug_diff_cam_seq_embed(data_sid, data_cid, data_id, select_k=2)
            aug_embed = self.concat_seq_embed((data_embed, other_embed_2))
            data_embed = self.concat_seq_embed((data_embed, other_embed_1))
        elif self.transform_type == "concat_multicam_reverse":
            other_embed_1 = self.aug_diff_cam_seq_embed(data_sid, data_cid, data_id, select_k=1)
            data_embed = self.concat_seq_embed((data_embed, other_embed_1))
            aug_embed = self.aug_reverse_seq_embed(data_embed)

        # Resize
        data_embed = self.resize_seq_len(data_embed)
        aug_embed = self.resize_seq_len(aug_embed)

        embeds = [
            torch.from_numpy(data_embed.copy()),
            torch.from_numpy(aug_embed.copy())
        ]
        
        high = self.get_pred_ratio() * self.seq_len
        mask =  np.hstack(
            [
                np.zeros(self.seq_len - int(high)), 
                np.ones(int(high)),
            ]
        ).astype(bool)
        np.random.shuffle(mask)

        masks = [torch.from_numpy(mask.copy())] * len(embeds)

        frames = torch.from_numpy(np.array(data_frames).copy())

        sample = {
            'name': data_name,
            'sid': data_sid,
            'cid': data_cid,
            'id': data_id,
            'frames':frames,
            'input': embeds,
            'mask': masks,
            'mask_num': int(high)
        }

        return sample

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root_dir = "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/all_train"
    debug_data = TrackletEmbeddingMask(
        root_dir, 
        seq_len=128, 
        pred_ratio=0.3, 
        pred_ratio_var=0, 
        pred_aspect_ratio=(0.3, 1/0.3), 
        pred_start_epoch=0, 
        transform_type="concat_multicam")
    # debug_data = TrackletEmbeddingInstance(root_dir, seq_len=128)
    debug_loader = DataLoader(debug_data, batch_size=4, shuffle=True)
    print(type(debug_loader))
    for epoch in range(0, 50):
        for iteration, data in enumerate(debug_loader):
            #embed = data['input']
            # mask = data['mask']
            #print(embed)
            # embed[mask,:] = 1
            # print(data["name"])
            print(data["sid"])
            print(data["cid"])
            print(data["id"])
            # print(data["input"].shape)
            # print(data["mask"])
            print(data["input"][0].shape)
            print(data["mask"][0].shape)
            # print(data["mask_num"])
            # print(data)

            ## Instance
            # feat, lab, _ = data
            # print(lab)
            # lab-=1
            # print(lab)

            print(f"min id: {debug_data.min_id}")
            print(f"max id: {debug_data.max_id}")

            print(len(debug_data))
            print(len(debug_loader))

            # print(embed[mask,:])

            print("...")



