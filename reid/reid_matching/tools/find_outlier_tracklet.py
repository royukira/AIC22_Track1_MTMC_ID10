#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   find_outlier_tracklet.py
@Time    :   2022/03/24 14:53:36
@Author  :   Roy Cheung 
@Version :   0.1
@Contact :   zhang.rui_sh@tslsmart.com
'''
# here put the import lib

import os
import cv2
import copy
import pickle
import math
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn import svm

from sklearn.preprocessing import scale
from os.path import join as opj

from cython_bbox import bbox_overlaps as bbox_ious

sys.path.append('../../../')
from config import cfg


C2C_ONEHOT = [
    (41,42), (42,43), (43,44), (44, 45), (45,46), 
    (46,45), (45,44), (44,43), (43,42), (42, 41)
]
C2C_ONEHOT_MAP = dict(zip(C2C_ONEHOT, range(len(C2C_ONEHOT))))

def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret

def calc_ious(atlwhs, btlwhs):
    """
    Compute cost based on IoU
    :type atlwhs: list[[xywh]]
    :type btlwhs: list[[xywh]]

    :rtype ious np.ndarray
    """
    atlwhs_np = np.array(atlwhs, dtype=np.float)
    btlwhs_np = np.array(btlwhs, dtype=np.float)

    atlbrs = np.zeros_like(atlwhs_np, dtype=np.float)
    btlbrs = np.zeros_like(btlwhs_np, dtype=np.float)
    for i in range(atlwhs_np.shape[0]):
        atlbrs[i] = tlwh_to_tlbr(atlwhs_np[i])

    for i in range(btlwhs_np.shape[0]):
        btlbrs[i] = tlwh_to_tlbr(btlwhs_np[i])

    ious = np.zeros((atlbrs.shape[0], btlbrs.shape[0]), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(atlbrs, btlbrs)

    return ious


def parse_pt(pt_file):
    with open(pt_file,'rb') as f:
        lines = pickle.load(f)
    img_rects = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:])
        tid = lines[line]['id']
        rect = list(map(lambda x: int(float(x)), lines[line]['bbox']))
        if fid not in img_rects:
            img_rects[fid] = list()
        rect.insert(0, tid)
        img_rects[fid].append(rect)
    return img_rects

def show_res(map_tid):
    show_dict = dict()
    for cid_tid in map_tid:
        iid = map_tid[cid_tid]
        if iid in show_dict:
            show_dict[iid].append(cid_tid)
        else:
            show_dict[iid] = [cid_tid]
    print(show_dict.keys())
    for k, v in show_dict.items():
        print('ID{}:{}'.format(k,v))


### calc time cost，and return c2c tuples of every tracklet
def calc_c2c_cost(tracklets:dict) -> dict:
    tid_cid_costs = {}
    for tid, cid_dict in tracklets.items():
        for cid in cid_dict.keys():
            if cid + 1 not in cid_dict:
                break
            current_fid_list = list(cid_dict[cid].keys())
            next_fid_list = list(cid_dict[cid+1].keys())

            current_fid_list.sort()
            next_fid_list.sort()

            key_name = None
            if next_fid_list[0] < current_fid_list[0]:
                time_diff = abs(current_fid_list[0] - next_fid_list[-1]) / 10.0
                key_name = (cid+1, cid)
            else:
                time_diff = abs(next_fid_list[0] - current_fid_list[-1]) / 10.0
                key_name = (cid, cid+1)

            if tid in tid_cid_costs:
                tid_cid_costs[tid].setdefault(key_name, time_diff)
            else:
                tid_cid_costs.setdefault(tid, {key_name:time_diff})

    tid_c2c = {}
    for tid, c2c_dict in tid_cid_costs.items():
        tid_c2c.setdefault(tid, list(c2c_dict.keys()))

    return tid_cid_costs, tid_c2c


def calc_c2c_speed(tracklets:dict) -> dict:
    ema_beta = 0.9
    tid_cid_speed = {}
    for tid, cid_dict in tracklets.items():
        for cid in cid_dict.keys():
            fid_list = list(cid_dict[cid].keys())
            fid_list.sort()

            rect_list = []
            for fid in fid_list:
                rect_list.append(cid_dict[cid][fid]['rect'])

            last_v = 0
            ema_speed = 0
            ema_accel = 0
            for i in range(len(rect_list)):
                if i == 0:
                    continue

                cur_rect = rect_list[i]
                prev_rect = rect_list[i-1]
                vx = cur_rect[0] - prev_rect[0]
                vy = cur_rect[1] - prev_rect[1]
                v = math.sqrt(vx**2 + vy**2)

                v_a = v - last_v
                last_v = v

                if v < 5:
                    continue

                if ema_speed == 0:
                    ema_speed = v
                else:
                    ema_speed = ema_beta * ema_speed + (1 - ema_beta) * v

                if ema_accel == 0:
                    ema_accel = v_a
                else:
                    ema_accel = ema_beta * ema_accel + (1 - ema_beta) * v_a   

            if tid in tid_cid_speed:
                tid_cid_speed[tid].setdefault(cid, (ema_speed, ema_accel))
            else:
                tid_cid_speed.setdefault(tid, {cid:(ema_speed, ema_accel)})
    return tid_cid_speed


def calc_c2c_traj_len(tracklets:dict) -> dict:
    tid_cid_traj_len = {}
    for tid, cid_dict in tracklets.items():
        for cid in cid_dict.keys():
            fid_list = list(cid_dict[cid].keys())
            fid_list.sort()
            
            rect_list = []
            for fid in fid_list:
                rect_list.append(cid_dict[cid][fid]['rect'])
            
            if tid in tid_cid_traj_len:
                tid_cid_traj_len[tid].setdefault(cid, {'len': len(fid_list), "rects": rect_list})
            else:
                tid_cid_traj_len.setdefault(tid, {cid:{'len':len(fid_list), "rects": rect_list}})
    return tid_cid_traj_len


def prepare_cluster_inputs(costs:dict, speeds:dict):
    tc_cost_speed = {}
    tid_list = list(costs.keys())
    #print(tid_list)
    for tid in tid_list:
        tc_cost_speed.setdefault(tid, {"c2c":[], "time":[], "speed":[]})
        cost_dict = costs[tid]
        speed_dict = speeds[tid]
        speed_keys_list = []
        for k, v in cost_dict.items():
            tc_cost_speed[tid]["c2c"].append(k)
            tc_cost_speed[tid]["time"].append(v)
            speed_keys_list.append(k[0])

        for sk in speed_keys_list:
            tc_cost_speed[tid]["speed"].append(speed_dict[sk])

    # [c2c_onehot_vec | cost speed acceleration]  -> shape of (1,13)
    tids = []
    tid_vecs = []
    tid_tc2c_dict = {}  
    for tid in tc_cost_speed.keys():
        num_tracklet = len(tc_cost_speed[tid]['c2c'])
        for i in range(num_tracklet):
            tvec = np.zeros((13), dtype=np.float) # vtec is the best!!!
            c2c = tc_cost_speed[tid]['c2c'][i]
            tvec[C2C_ONEHOT_MAP[c2c]] = 1.0 
            tvec[10] = tc_cost_speed[tid]['time'][i]
            tvec[11] = tc_cost_speed[tid]['speed'][i][0] # speed
            tvec[12] = tc_cost_speed[tid]['speed'][i][1] # acceleration
            tid_vecs.append(tvec)
            tids.append(tid)
            tid_tc2c_dict.setdefault(len(tids), (tid, c2c))

    tid_idx_vec = np.zeros(
        (
            len(tids), 
            max(list(tc_cost_speed.keys()))
        )
    )
    for i in range(len(tids)):
        tid_idx_vec[i, tids[i]-1] = 1

    tid_vecs = np.vstack(tid_vecs)
    
    return tid_vecs, tid_idx_vec, tid_tc2c_dict


def find_speed_outliers(speed_dict:dict, traj_len_dict:dict) -> dict:
    error_tid_speed = {}
    for tid, speed_seq in speed_dict.items():
        is_all_zero_error = False
        cid_list = list(speed_seq.keys())
        cid_list.sort()
        for cid in cid_list:
            if is_all_zero_error:
                error_tid_speed[tid].append(cid)
                continue      
            
            if speed_seq[cid] == (0, 0) and cid != max(cid_list):
                if traj_len_dict[tid][cid]['len'] > 10: 
                    is_all_zero_error = True
                if tid in error_tid_speed:
                    error_tid_speed[tid].append(cid)
                else:
                    error_tid_speed.setdefault(tid, [cid])
                    
    return error_tid_speed


def find_direct_outliers(cost_dict:dict) -> dict:
    error_tid_c2c = {}
    for tid, time_seq in cost_dict.items():
        is_conflit = False
        to_41 = False
        to_46 = False
        to_41_cnt = 0
        to_46_cnt = 0
        tmp_ec2c_list = []
        tmp_ec2c_costs_list = []
        c2c_list = list(time_seq.keys())
        for i in range(len(c2c_list)):
            c2c = c2c_list[i]
            if i == 0:
                to_41 = c2c[0] > c2c[1]
                to_46 = c2c[0] < c2c[1]
                if to_41:
                    to_41_cnt += 1
                elif to_46:
                    to_46_cnt += 1
                continue

            if (to_41 and c2c[0] < c2c[1]) or (to_46 and c2c[0] > c2c[1]):
                if i == 1:
                    tmp_ec2c_list.append(c2c_list[0])
                    tmp_ec2c_costs_list.append(time_seq[c2c_list[0]]) 
                tmp_ec2c_list.append(c2c)
                tmp_ec2c_costs_list.append(time_seq[c2c])
                is_conflit = True

            to_41 = c2c[0] > c2c[1]
            to_46 = c2c[0] < c2c[1]
            if to_41:
                to_41_cnt += 1
            elif to_46:
                to_46_cnt += 1

        if is_conflit:
            c2c_main_direct = 0 # (to)41 or (to)46 or (Unknown)0
            if to_41_cnt > to_46_cnt:
                c2c_main_direct = 41
            elif to_46_cnt > to_41_cnt:
                c2c_main_direct = 46

            error_tid_c2c.setdefault(tid, {"c2c":tmp_ec2c_list, "cost": tmp_ec2c_costs_list, "direct":c2c_main_direct})
    return error_tid_c2c


def find_overshort_outliers(traj_len_dict:dict, only_find_one_cam=False) -> dict:
    error_tid_traj = {}
    for tid, traj_len_seq in traj_len_dict.items():
        cid_list = list(traj_len_seq.keys())
        cid_list.sort()
        if only_find_one_cam:
            if len(cid_list) == 1:
                print(f"{tid} only has one cam")
                if tid in error_tid_traj:
                    error_tid_traj[tid].setdefault(cid_list[0], traj_len_seq[cid_list[0]]['len'])
                else:
                    error_tid_traj.setdefault(tid, {cid_list[0]:traj_len_seq[cid_list[0]]['len']})
        else: 
            for cid in cid_list:
                if traj_len_seq[cid]['len'] < 10:
                    if tid in error_tid_traj:
                        error_tid_traj[tid].setdefault(cid, traj_len_seq[cid]['len'])
                    else:
                        error_tid_traj.setdefault(tid, {cid:traj_len_seq[cid]['len']})
    return error_tid_traj


def find_cluster_outliers(data, idx, n_clusters, is_scale=True):
    # 指定聚类个数，准备进行数据聚类
    kmeans = KMeans(n_clusters=n_clusters)
    # 用于存储聚类相关的结果
    outlier_res = []
    # 判断是否需要对数据做标准化处理
    if is_scale:
        std_data = scale(data[:,10:], axis=0) # 标准化
        data = np.concatenate((data[:,:10], std_data), axis=1)
    kmeans.fit(data)  # 聚类拟合
    # 返回簇标签
    labels = kmeans.labels_
    # 返回簇中心
    centers = kmeans.cluster_centers_
    for label in set(labels):
        # 计算簇内样本点与簇中心的距离
        diff = data[np.array(labels)==label,] - np.array(centers[label])
        dist = np.sum(np.square(diff), axis=1)
        # 计算判断异常的阈值
        UL = dist.mean() + 3*dist.std()
        # UL = dist.mean() + 3.5*dist.std()
        # 识别异常值，1表示异常，0表示正常
        OutLine = np.where(dist > UL, 1, 0)       
        # 找到异常tid以及对应的c2c
        track_idx = np.arange(idx.shape[0])[(np.array(labels) == label)]
        data_idx = idx[(np.array(labels) == label),]
        outlier_stids = track_idx[OutLine==1,]
        outlier_tids = data_idx[OutLine==1,]
        outlier_res.append((np.where(outlier_tids==1)[1] + 1, outlier_stids + 1))
    return outlier_res


def find_cluster_outliers_2(data, idx, n_clusters, is_scale=True, method="kmeans"):
    if method ==  "kmeans":
        return find_cluster_outliers(data, idx, n_clusters, is_scale)
    elif method == "svm":
        ocs = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
        outlier_res = []
        if is_scale:
            std_data = scale(data[:,10:], axis=0)
            data = np.concatenate((data[:,:10], std_data), axis=1)
        data_pred = ocs.fit_predict(data)
        OutLine = np.where(data_pred == -1, 1, 0)
        track_idx = np.arange(idx.shape[0])
        data_idx = idx
        outlier_stids = track_idx[OutLine==1,]
        outlier_tids = data_idx[OutLine==1,]
        outlier_res.append((np.where(outlier_tids==1)[1] + 1, outlier_stids + 1))
        return outlier_res
    else:
        raise AssertionError("only kmeans or svm")


def fix_overshort_outlier(error_tid_traj_len:dict, tid_cid_dict:dict):
    temp_tid_cid = copy.deepcopy(tid_cid_dict)
    for e_tid, error_cids in error_tid_traj_len.items():
        for e_cid in error_cids:
            if len(error_cids) == 1:
                is_del_all = False
            else:
                is_del_all = True
            if error_cids[e_cid] < 3:
                del temp_tid_cid[e_tid][e_cid]
                continue
            
            e_traj = temp_tid_cid[e_tid][e_cid]
            e_fid_list = list(e_traj.keys())
            e_fid_list.sort()
            start_frame = e_fid_list[0]
            start_box = e_traj[start_frame]['rect']
            candidate_boxes = []
            candidate_tids = []

            ### Fine the broken traj. candidate
            for tid, cid_dict in temp_tid_cid.items():
                if tid == e_tid:
                    continue
                for cid in cid_dict.keys():
                    if cid != e_cid:
                        continue
                    if (start_frame - 1) in cid_dict[cid]:
                        candidate_boxes.append(cid_dict[cid][start_frame - 1]['rect'])
                        candidate_tids.append(tid)
            
            if len(candidate_boxes) == 0:
                print(f"{e_tid}-{e_cid} cannot find candidate boxes")
                continue

            ### Calc iou with the candidate box in last frame and retrive the broken traj.
            ious = calc_ious([start_box], candidate_boxes)
            nearest_idx = np.argmax(ious)
            if ious[:,nearest_idx] > 0.2:
                conflit_cnt = 0
                right_tid = candidate_tids[nearest_idx]
                # Firstly, move the wrong traj. to the right_tid
                for e_fid, outs in e_traj.items():
                    if e_fid in temp_tid_cid[right_tid][e_cid]:
                        print(f"{e_tid}-{e_cid}-{e_fid} x-> {right_tid}-{e_cid}-{e_fid}")
                        conflit_cnt += 1
                        continue
                    if len(error_cids) == 1:
                        is_del_all = True
                    # Change TID
                    output_split = outs['output'].split(" ")
                    output_split[1] = f"{right_tid}"
                    outs['output'] = " ".join(output_split)
                    temp_tid_cid[right_tid][e_cid].setdefault(e_fid, outs)
                if is_del_all:
                    print(f"{e_tid} delete {e_cid}")
                    del temp_tid_cid[e_tid][e_cid]
                # Secondly, move the trajs of the wrong tid in the following cameras to right_tid (only add, not modified any traj in right_tid)
                is_break = False
                del_cids = []
                for cid in temp_tid_cid[e_tid].keys():
                    if (cid < e_cid) or is_break:
                        # Don't move the trajs. in the former cameras 
                        continue
                    if (cid == e_cid + 1) and (cid in temp_tid_cid[right_tid]):
                        # and the trajs of right_tid in the following cameras cannot be modified
                        is_break = True
                        continue
                    for fid, outs in temp_tid_cid[e_tid][cid].items():
                        # Change TID
                        output_split = outs['output'].split(" ")
                        output_split[1] = f"{right_tid}"
                        outs['output'] = " ".join(output_split)
                        if cid in temp_tid_cid[right_tid]:
                            temp_tid_cid[right_tid][cid].setdefault(fid, outs)
                        else:
                            temp_tid_cid[right_tid].setdefault(cid, {fid:outs})
                    del_cids.append(cid)
                # Finally, delete the rest 
                if is_del_all:
                    for dc in del_cids:
                        print(f"{e_tid} delete {dc}")
                        del temp_tid_cid[e_tid][dc]

    return temp_tid_cid


def del_speed_outlier(error_tid_speed:dict, tid_cid_dict:dict):
    temp_tid_cid = copy.deepcopy(tid_cid_dict)
    for e_tid, error_cids in error_tid_speed.items():
        for e_cid in error_cids:
            del temp_tid_cid[e_tid][e_cid]
    return temp_tid_cid


def del_direct_outlier(error_tid_c2c:dict, tid_cid_dict:dict):
    temp_tid_cid = copy.deepcopy(tid_cid_dict)
    for e_tid, error in error_tid_c2c.items():
        e_c2c_list = error["c2c"]
        e_c2c_costs_list = error["cost"]
        e_direct = error["direct"]
        if e_direct != 0:
            for e_c2c_idx in range(len(e_c2c_list)):
                e_c2c = e_c2c_list[e_c2c_idx]
                if e_c2c[0] < e_c2c[1] and e_direct == 41:
                    if len(e_c2c_list) == 1:
                        del temp_tid_cid[e_tid][e_c2c[1]]
                    else:
                        del temp_tid_cid[e_tid][e_c2c[0]]
                elif e_c2c[0] > e_c2c[1] and e_direct == 46:
                    if len(e_c2c_list) == 1:
                        print(f"{e_tid} delete {e_c2c[0]}")
                        del temp_tid_cid[e_tid][e_c2c[0]]
                    else:
                        print(f"{e_tid} delete {e_c2c[1]}")
                        del temp_tid_cid[e_tid][e_c2c[1]]
        else:
            inter_cam = list(set(e_c2c_list[0]) & set(e_c2c_list[1]))[0]
            max_cost = max(e_c2c_costs_list)
            if max_cost > 150:  #  1500 / 10
                del_idx = e_c2c_costs_list.index(max_cost)
                del_e_c2c = e_c2c_list[del_idx]
                if del_e_c2c[0] != inter_cam:
                    print(f"{e_tid} delete {del_e_c2c[0]}")
                    del temp_tid_cid[e_tid][del_e_c2c[0]]
                elif del_e_c2c[1] != inter_cam: 
                    print(f"{e_tid} delete {del_e_c2c[1]}")
                    del temp_tid_cid[e_tid][del_e_c2c[1]]
    return temp_tid_cid


def del_cluster_outlier(outlier, tid_stid_dict, tid_c2c_dict, tid_cid_dict):
    temp_tid_cid = copy.deepcopy(tid_cid_dict)
    outlier_tid_stid = []
    for i in list(outlier[0][1]):
        outlier_tid_stid.append(tid_stid_dict[i])

    for e_tid_c2c in outlier_tid_stid:
        e_tid = e_tid_c2c[0]
        e_c2c = e_tid_c2c[1]

        c2c_list = list(tid_c2c_dict[e_tid])
        print(f"{e_tid}:{c2c_list} -- del {e_c2c}, idx: {c2c_list.index(e_c2c)}")

        # 若只有一条跨镜头轨迹, 全删
        if len(c2c_list) == 1:
            del temp_tid_cid[e_tid][c2c_list[0][0]]
            del temp_tid_cid[e_tid][c2c_list[0][1]]
        else:
            e_idx = c2c_list.index(e_c2c)
            keep_cams = []
            if e_idx - 1 >=0:
                prev_c2c = c2c_list[e_idx-1]
                if e_c2c[0] in prev_c2c:
                    keep_cams.append(e_c2c[0])
                elif e_c2c[1] in prev_c2c:
                    keep_cams.append(e_c2c[1])
            if e_idx + 1 < len(c2c_list):
                next_c2c = c2c_list[e_idx+1]
                if e_c2c[0] in next_c2c:
                    keep_cams.append(e_c2c[0])
                elif e_c2c[1] in next_c2c:
                    keep_cams.append(e_c2c[1])

            if e_c2c[0] not in keep_cams:
                del temp_tid_cid[e_tid][e_c2c[0]]
                print(f"{e_tid}:del cam {e_c2c[0]}")
            if e_c2c[1] not in keep_cams:
                del temp_tid_cid[e_tid][e_c2c[1]]
                print(f"{e_tid}:del cam {e_c2c[1]}")

    return temp_tid_cid


def renew_tracklets(tid_cid_dict: dict):
    # 要改output的id
    new_tid_cid_dict = {}
    tracklet_new_id = 1
    for tid, cid_seq in tid_cid_dict.items():
        if cid_seq == {}:
            continue
        for cid, fid_seq in cid_seq.items():
            for fid, outs in fid_seq.items():
                output_split = outs['output'].split(" ")
                output_split[1] = f"{tracklet_new_id}"
                new_output = " ".join(output_split)
                if tracklet_new_id in new_tid_cid_dict:
                    if cid in new_tid_cid_dict[tracklet_new_id]:
                        new_tid_cid_dict[tracklet_new_id][cid].setdefault(fid, {'output':new_output})
                    else:
                        new_tid_cid_dict[tracklet_new_id].setdefault(cid, {fid:{'output':new_output}})
                else:
                    new_tid_cid_dict.setdefault(tracklet_new_id, {cid:{fid:{'output':new_output}}})
        tracklet_new_id+=1
    return new_tid_cid_dict

# def renew_tracklets(tid_cid_dict: dict):
#     new_tid_cid_dict = {}
#     tracklet_cnt = 1
#     for k, v in tid_cid_dict.items():
#         if v == {}:
#             continue
#         new_tid_cid_dict.setdefault(tracklet_cnt, v)
#         tracklet_cnt+=1
#     return new_tid_cid_dict


if __name__ == "__main__":
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()

    output_txt = os.path.join(cfg.OUTPUT_DIR, cfg.MCMT_OUTPUT_TXT)
    cluster_path = os.path.join(cfg.OUTPUT_DIR,'test_cluster.pkl')
    data_dir = cfg.DATA_DIR
    roi_dir = os.path.join(cfg.CHALLENGE_DATA_DIR, 'S06')
    mot_dir = os.path.join(data_dir, 'mot_result')

    map_tid = pickle.load(open(cluster_path, 'rb'))['cluster']
    cam_paths = os.listdir(mot_dir)
    cam_paths = list(filter(lambda x: 'c' in x, cam_paths))
    cam_paths.sort()

    # Preprocessing multi-cam tracklet cluster data
    tid_cid_dict = {}
    for cam_path in cam_paths:
        cid = int(cam_path.split('.')[0][-3:])
        roi = cv2.imread(opj(roi_dir, '{}/roi.jpg'.format(cam_path.split('.')[0][-4:])), 0)
        height, width = roi.shape
        img_rects = parse_pt(opj(mot_dir, cam_path,'{}_mot_feat_break.pkl'.format(cam_path)))
        for fid in img_rects:
            tid_rects = img_rects[fid]
            fid = int(fid)+1
            for tid_rect in tid_rects:
                tid = tid_rect[0]
                # if (cid, tid) == (41, 252):
                #     input("...")
                rect = tid_rect[1:]
                cx = 0.5*rect[0] + 0.5*rect[2]
                cy = 0.5*rect[1] + 0.5*rect[3]
                w = rect[2] - rect[0]
                w = min(w*1.2,w+40)
                h = rect[3] - rect[1]
                h = min(h*1.2,h+40)
                rect[2] -= rect[0]
                rect[3] -= rect[1]
                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                x1, y1 = max(0, cx - 0.5*w), max(0, cy - 0.5*h)
                x2, y2 = min(width, cx + 0.5*w), min(height, cy + 0.5*h)
                w , h = x2-x1 , y2-y1

                new_rect = list(map(int, [x1, y1, w, h]))
                # new_rect = rect # 使用原bbox
                rect = list(map(int, rect))
                if (cid, tid) in map_tid:
                    new_tid = map_tid[(cid, tid)]
                    output = str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' + ' '.join(map(str, new_rect)) + ' -1 -1' '\n'
                    if new_tid in tid_cid_dict:
                        if cid in tid_cid_dict[new_tid]:
                            tid_cid_dict[new_tid][cid].setdefault(fid, {'rect':new_rect, 'output':output})
                        else:
                            tid_cid_dict[new_tid].setdefault(cid, {fid:{'rect':new_rect, 'output':output}})
                    else:
                        tid_cid_dict.setdefault(new_tid, {cid:{fid:{'rect':new_rect, 'output':output}}})
    
    
    """
    ### 过滤规则
    * 速度及加速度异常:非最后cam的tracklet若出现速度和加速度均为0的情况,则认为是异常情况,且后面的所有轨迹都被删除,因为出现速度和加速度均为0一般是等红灯,若到视频结束，依旧未动，车辆不可能在下一个相邻cam出现；若出现速度为0，加速度非0情况，也认为是异常情况；
    * 检测方向冲突：同一`fid`下的time序列,是否同时出现`(cid, cid+1),(cid+2, cid+1)...`这种情况，正常情况是`(cid,cid+1),(cid+1,cid+2)`
    * 聚类+异常值分析:结合相邻cam one hot,耗时,速度,加速度分析
    * 轨迹长度过短
    * 单一镜头轨迹
    """
    print(f"num of old tracklets: {len(tid_cid_dict)}")

    # Calc cam2cam time cost & (speed, acceleration)
    tid_cid_traj_len = calc_c2c_traj_len(tid_cid_dict)
    tid_cid_costs, tid_c2cs = calc_c2c_cost(tid_cid_dict)
    tid_cid_speeds = calc_c2c_speed(tid_cid_dict)
    
    ## speed check
    error_tid_speed = find_speed_outliers(tid_cid_speeds, tid_cid_traj_len)
    print(f"speed outliers: {error_tid_speed}")
    # delete outliers
    tid_cid_dict = del_speed_outlier(error_tid_speed, tid_cid_dict)
    # update
    tid_cid_costs, tid_c2cs = calc_c2c_cost(tid_cid_dict)
    tid_cid_speeds = calc_c2c_speed(tid_cid_dict)
    #print(f"speed outliers after filtering:{error_tid_speed}")

    ## direction check
    error_tid_c2c = find_direct_outliers(tid_cid_costs)
    print(f"direction outliers:{error_tid_c2c}")
    ## delete outliers
    tid_cid_dict = del_direct_outlier(error_tid_c2c, tid_cid_dict)
    ## update
    tid_cid_costs, tid_c2cs = calc_c2c_cost(tid_cid_dict)
    tid_cid_speeds = calc_c2c_speed(tid_cid_dict)
    #print(f"direction outliers after filtering:{error_tid_c2c}")

    ## Prepare data
    tid_vecs, tid_idx_vec, tid_stid_dict = prepare_cluster_inputs(tid_cid_costs, tid_cid_speeds)
    ## find outliers via Kmeans
    outlier = find_cluster_outliers(tid_vecs, tid_idx_vec, 2, False)
    # outlier = find_cluster_outliers(tid_vecs, tid_idx_vec, 2, False)
    # outlier = find_cluster_outliers_2(tid_vecs, tid_idx_vec, 1, False, method="kmeans")    # kmeans or svm
    outlier_tid_stid = []
    for i in list(outlier[0][1]):
        outlier_tid_stid.append(tid_stid_dict[i])
    print(outlier_tid_stid)
    # # # ## delete outliers
    tid_cid_dict = del_cluster_outlier(outlier, tid_stid_dict, tid_c2cs, tid_cid_dict)

     ## Firstly, check single-camera traj. len
    tid_cid_traj_len = calc_c2c_traj_len(tid_cid_dict)
    error_tid_traj_len = find_overshort_outliers(tid_cid_traj_len)
    print(f"traj len outliers: {error_tid_traj_len}")
    # fix the outlier
    tid_cid_dict = fix_overshort_outlier(error_tid_traj_len, tid_cid_dict)

    # find only one cam
    tid_cid_traj_len = calc_c2c_traj_len(tid_cid_dict)
    error_tid_traj_len = find_overshort_outliers(tid_cid_traj_len, only_find_one_cam=True)
    print(f"traj len outliers: {error_tid_traj_len}")
    # fix the outlier
    tid_cid_dict = fix_overshort_outlier(error_tid_traj_len, tid_cid_dict)

    
    ## Clean empty tracklets
    new_tracklets = renew_tracklets(tid_cid_dict)
    print(f"num of new tracklets: {len(new_tracklets)}")

    # print(new_tracklets.keys())
    #print(new_tracklets[228])

    ## Output
    f_w = open(output_txt, 'w')
    for tid, cid_dict in new_tracklets.items():
        for cid, fid_dict in cid_dict.items():
            for fid, outs in fid_dict.items():
                f_w.write(outs['output'])
    f_w.close()
