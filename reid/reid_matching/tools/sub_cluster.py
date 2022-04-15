from re import sub
from utils.filter import *
from utils.visual_rr import visual_rerank
from sklearn.cluster import AgglomerativeClustering
import sys
sys.path.append('../../../')
from config import cfg
from reid.reid_matching.tools import find_outlier_tracklet as fot

import copy

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

def parse_tid_cid_rect(cid_tid_label, cam_img_rects, cam_img_rois):
    # print(cid_tid_label)
    # input("...")
    tid_cid_rects_dict = {}
    newtid_cid_oldtid_dict = {}
    for cid, img_rects in cam_img_rects.items():
        roi = cam_img_rois[cid]
        height, width = roi.shape
        cid = int(cid.split("c")[-1])
        for fid in img_rects:
            tid_rects = img_rects[fid]
            fid = int(fid)+1
            for tid_rect in tid_rects:
                tid = tid_rect[0]
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
                if (cid, tid) in cid_tid_label:
                    new_tid = cid_tid_label[(cid, tid)] 

                    if new_tid in tid_cid_rects_dict:
                        if cid in tid_cid_rects_dict[new_tid]:
                            tid_cid_rects_dict[new_tid][cid].setdefault(fid, {'rect':new_rect, 'cid_tid':(cid, tid)})
                     
                        else:
                            tid_cid_rects_dict[new_tid].setdefault(cid, {fid:{'rect':new_rect, 'cid_tid':(cid, tid)}})
                    else:
                        tid_cid_rects_dict.setdefault(new_tid, {cid:{fid:{'rect':new_rect, 'cid_tid':(cid, tid)}}})

                    if new_tid in newtid_cid_oldtid_dict:
                        if cid in newtid_cid_oldtid_dict[new_tid]:
                            continue
                        newtid_cid_oldtid_dict[new_tid].setdefault(cid, tid)
                    else:
                        newtid_cid_oldtid_dict.setdefault(new_tid, {cid:tid})

    return tid_cid_rects_dict, newtid_cid_oldtid_dict


def get_sim_matrix(_cfg,cid_tid_dict,cid_tids,c2c,conflit_tids=[], is_norm=True):
    count = len(cid_tids)
    print('count: ', count)
    print(c2c)

    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    print(q_arr.shape)
    print(g_arr.shape)

    if is_norm:
        q_arr = normalize(q_arr, axis=1)
        g_arr = normalize(g_arr, axis=1)

    # st mask
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)                # filter same cam
    st_mask = st_filter(st_mask, cid_tids, cid_tid_dict)

    # # conflit mask ()
    if len(conflit_tids) > 0:
        st_mask, remain_conflit_tids, conflit_idx_pairs = conflit_mot_ignore(st_mask, cid_tids, c2c, conflit_tids)  # filter conflit mot
    else:
        remain_conflit_tids = []

    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, _cfg)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask
    print(sim_matrix)

    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix, remain_conflit_tids


def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def combin_cluster(sub_labels,cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    temp_cid_tid_dict = copy.deepcopy(cid_tid_dict)
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: 
            continue
        mean_feat = np.array([temp_cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            temp_cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return temp_cid_tid_dict

def get_labels(_cfg, cid_tid_dict, cid_tids, score_thr):
    print(cid_tid_dict.keys())
    # print(cid_tid_dict)
    # input("...")

    # 1st cluster
    sub_cid_tids = subcam_list(cid_tid_dict,cid_tids)   # filter tracklets which are not in main road
    print(sub_cid_tids.keys())
    # input("...")
    sub_labels = dict()
    dis_thrs = [0.7,0.5,0.5,0.5,0.5,
                0.7,0.5,0.5,0.5,0.5]
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix, _ = get_sim_matrix(_cfg,cid_tid_dict,sub_cid_tids[sub_c_to_c],sub_c_to_c)
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        # print(cluster_labels)   # [label0, label1, ...]
        # input("...")
        label_idxs = get_match(cluster_labels)  # [[label0_idx0, label0_idx1...],[]]
        # print(label_idxs)
        # input("...")
        cluster_cid_tids = get_cid_tid(label_idxs,sub_cid_tids[sub_c_to_c])     # [[label0_cid_tid0, label0_cid_tid1,...],...] 
        # print(cluster_cid_tids)
        # input("...")
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    label_idxs,sub_cluster = combin_cluster(sub_labels,cid_tids)
    # print(label_idxs)
    # input("...")

    # 2nd cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
    print(sub_cid_tids.keys())
    sub_labels = dict()
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix, _ = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c],sub_c_to_c)
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-0.1, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        label_idxs = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(label_idxs,sub_cid_tids[sub_c_to_c])
        # print(cluster_cid_tids)
        # input("...")
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    label_idxs,sub_cluster = combin_cluster(sub_labels,cid_tids)

    return label_idxs

def get_labels_2(_cfg, cid_tid_dict, cid_tids, cam_img_rects, cam_img_rois, score_thr):
    t = 0
    epoch = 2
    num_new_outliers = 0
    total_outliers = []
    while(t < epoch):
        if (num_new_outliers == 0 and t > 0):
            break
        outliers = copy.deepcopy(total_outliers)
        # 1st cluster
        print("1st cluster...")
        sub_cid_tids = subcam_list(cid_tid_dict,cid_tids)   # filter tracklets which are not in main road
        sub_labels = dict()
        dis_thrs = [0.7,0.5,0.5,0.5,0.5,
                    0.7,0.5,0.5,0.5,0.5]
        for i,sub_c_to_c in enumerate(sub_cid_tids):
            sim_matrix, outliers = get_sim_matrix(_cfg,cid_tid_dict,sub_cid_tids[sub_c_to_c],sub_c_to_c,conflit_tids=outliers, is_norm=False)
            cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                        linkage='complete').fit_predict(1 - sim_matrix) # farthest distance
            cid_tids_idxs = get_match(cluster_labels)  # [[label0_idx0, label0_idx1...],[]]
            cluster_cid_tids = get_cid_tid(cid_tids_idxs,sub_cid_tids[sub_c_to_c])     # [[label0_cid_tid0, label0_cid_tid1,...],...] 
            sub_labels[sub_c_to_c] = cluster_cid_tids
        print("old tricklets:{}".format(len(cid_tids)))
        cid_tids_idxs,sub_cluster = combin_cluster(sub_labels,cid_tids)
        # print(label_idxs)
        # input("...")

        # 2nd cluster
        print("2st cluster...")
        cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
        sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
        # print(sub_cid_tids.keys())
        sub_labels = dict()
        for i,sub_c_to_c in enumerate(sub_cid_tids):
            sim_matrix, outliers = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c],sub_c_to_c,conflit_tids=outliers, is_norm=False)
            cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-0.1, affinity='precomputed',
                                        linkage='complete').fit_predict(1 - sim_matrix)
            cid_tids_idxs = get_match(cluster_labels)
            cluster_cid_tids = get_cid_tid(cid_tids_idxs,sub_cid_tids[sub_c_to_c])
            # print(cluster_cid_tids)
            # input("...")
            sub_labels[sub_c_to_c] = cluster_cid_tids
        print("old tricklets:{}".format(len(cid_tids)))
        cid_tids_idxs, sub_cluster = combin_cluster(sub_labels,cid_tids)

        ## =========================================
        ##            Anormaly Detection           =
        ## =========================================
        outliers.clear()
        single_cid_tids_idx = []
        new_cid_tids_idx = []
        for ct_list in cid_tids_idxs:
            if len(ct_list) <= 1: 
                single_cid_tids_idx.append(cid_tids[ct_list[0]])
                continue
            if len(ct_list)!=len(set(ct_list)): continue
            new_cid_tids_idx.append([cid_tids[c] for c in ct_list])
        cid_tid_label = dict()
        for i, c_list in enumerate(new_cid_tids_idx):
            for c in c_list:
                cid_tid_label[c] = i + 1
        
        tid_cid_rects_dict, newtid_cid_oldtid_dict = parse_tid_cid_rect(cid_tid_label, cam_img_rects, cam_img_rois)
        
        ## calc single-camera traj. len
        tid_cid_traj_len = fot.calc_c2c_traj_len(tid_cid_rects_dict)

        # speed outlier
        tid_cid_speeds = fot.calc_c2c_speed(tid_cid_rects_dict)
        speed_outlier = fot.find_speed_outliers(tid_cid_speeds, tid_cid_traj_len)
        print(speed_outlier)
        
        for e_tid, e_cids in speed_outlier.items():
            temp_outliers = []
            e_cids.sort()
            for eidx in range(len(e_cids)):
                if eidx > 1:
                    break
                e_cid = e_cids[eidx]
                temp_outliers.append(
                    (e_cid, newtid_cid_oldtid_dict[e_tid][e_cid])
                )
                del tid_cid_rects_dict[e_tid][e_cid]                 
            for i in range(len(temp_outliers) - 1):
                outliers.append([temp_outliers[i], temp_outliers[i+1]])

        # print(outliers)
        # input("...")

        # direction outlier
        tid_cid_costs, _ = fot.calc_c2c_cost(tid_cid_rects_dict)
        direct_outlier = fot.find_direct_outliers(tid_cid_costs)
        print(direct_outlier)
        for e_tid, error in direct_outlier.items():
            e_c2c_list = error["c2c"]
            e_direct = error["direct"]
            e_c2c_costs_list = error["cost"]
            if e_direct == 0:
                inter_cam = list(set(e_c2c_list[0]) & set(e_c2c_list[1]))[0]
                max_cost = max(e_c2c_costs_list)
                if max_cost > 150:  #  1500 / 10
                    temp_outliers = []
                    del_idx = e_c2c_costs_list.index(max_cost)
                    del_e_c2c = e_c2c_list[del_idx]
                    temp_outliers.append(
                        (del_e_c2c[0], newtid_cid_oldtid_dict[e_tid][del_e_c2c[0]])
                    )
                    temp_outliers.append(
                        (del_e_c2c[1], newtid_cid_oldtid_dict[e_tid][del_e_c2c[1]])
                    )
                    if del_e_c2c[0] != inter_cam:
                        del tid_cid_rects_dict[e_tid][del_e_c2c[0]]
                    elif del_e_c2c[1] != inter_cam: 
                        del tid_cid_rects_dict[e_tid][del_e_c2c[1]]
            else:
                for e_c2c in e_c2c_list:
                    if e_c2c[0] < e_c2c[1] and e_direct == 41:
                        temp_outliers = []
                        temp_outliers.append(
                            (e_c2c[0], newtid_cid_oldtid_dict[e_tid][e_c2c[0]])
                        )
                        temp_outliers.append(
                            (e_c2c[1], newtid_cid_oldtid_dict[e_tid][e_c2c[1]])
                        )
                        outliers.append(temp_outliers)
                        if len(e_c2c_list) == 1:
                            del tid_cid_rects_dict[e_tid][e_c2c[1]]
                        else:
                            del tid_cid_rects_dict[e_tid][e_c2c[0]]
                    elif e_c2c[0] > e_c2c[1] and e_direct == 46:
                        temp_outliers = []
                        temp_outliers.append(
                            (e_c2c[0], newtid_cid_oldtid_dict[e_tid][e_c2c[0]])
                        )
                        temp_outliers.append(
                            (e_c2c[1], newtid_cid_oldtid_dict[e_tid][e_c2c[1]])
                        )
                        outliers.append(temp_outliers)
                        if len(e_c2c_list) == 1:
                            del tid_cid_rects_dict[e_tid][e_c2c[0]]
                        else:
                            del tid_cid_rects_dict[e_tid][e_c2c[1]]

        #print(outliers)
        #input("...")

        # Oneclass cluster 
        tid_cid_costs, _ = fot.calc_c2c_cost(tid_cid_rects_dict)
        tid_cid_speeds = fot.calc_c2c_speed(tid_cid_rects_dict)
        tid_vecs, tid_idx_vec, tid_stid_dict = fot.prepare_cluster_inputs(tid_cid_costs, tid_cid_speeds)
        cluster_outlier = fot.find_cluster_outliers_2(tid_vecs, tid_idx_vec, 2, False, method="kmeans")
        outlier_tid_stid = []
        for i in list(cluster_outlier[0][1]):
            outlier_tid_stid.append(tid_stid_dict[i])
        print(outlier_tid_stid)
        for e_tid_c2c in outlier_tid_stid:
            e_tid = e_tid_c2c[0]
            e_c2c = e_tid_c2c[1]

            temp_outliers = []
            temp_outliers.append(
                (e_c2c[0], newtid_cid_oldtid_dict[e_tid][e_c2c[0]])
            )
            temp_outliers.append(
                (e_c2c[1], newtid_cid_oldtid_dict[e_tid][e_c2c[1]])
            )
            outliers.append(temp_outliers)

        print(outliers)
        num_old_outliers = len(total_outliers)
        total_outliers.extend(outliers)
        total_outliers = list(set(tuple(_) for _ in total_outliers))
        print(total_outliers)
        num_new_outliers = len(total_outliers) - num_old_outliers
        print(f"Num of outliers: {len(total_outliers)}")
        print(f"new add: {num_new_outliers}")
        
        t += 1
        print("===============================================================")
        
    return cid_tids_idxs

if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    scene_name = ['S06']
    # scene_cluster = [[41, 42, 43, 44, 45, 46]]
    scene_cluster = cfg.CAM_CLUSTER
    
    data_dir = cfg.DATA_DIR
    mot_dir =os.path.join(data_dir, 'mot_result')
    cam_paths = os.listdir(mot_dir)
    cam_paths = list(filter(lambda x: 'c' in x, cam_paths))
    cam_paths.sort()
    mot_feat_break_paths = [os.path.join(mot_dir, c, '{}_mot_feat_break.pkl'.format(c)) for c in cam_paths]
    img_rects = [parse_pt(p) for p in mot_feat_break_paths]
    cam_img_rects = dict(zip(cam_paths, img_rects))

    roi_dir = cfg.ROI_DIR
    rois = [cv2.imread(os.path.join(roi_dir, '{}/roi.jpg'.format(c.split('.')[0][-4:])), 0) for c in cam_paths]
    cam_img_rois = dict(zip(cam_paths, rois))
    
    out_dir = cfg.OUTPUT_DIR
    fea_dir = f'{out_dir}/trajectory/'
    cluster_path = f'{out_dir}/test_cluster.pkl'

    cid_tid_dict = dict()

    for pkl_path in os.listdir(fea_dir):
        cid = int(pkl_path.split('.')[0][-3:])
        with open(opj(fea_dir, pkl_path),'rb') as f:
            lines = pickle.load(f)
        for line in lines:
            tracklet = lines[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster[0]])
    #clu = get_labels(cfg, cid_tid_dict, cid_tids, score_thr=cfg.SCORE_THR)
    clu = get_labels_2(cfg, cid_tid_dict, cid_tids, cam_img_rects, cam_img_rois, score_thr=cfg.SCORE_THR)
    print('all_clu:', len(clu))
    new_clu = list()
    for c_list in clu:
        if len(c_list) <= 1: continue
        cam_list = [cid_tids[c][0] for c in c_list]
        if len(cam_list)!=len(set(cam_list)): continue
        new_clu.append([cid_tids[c] for c in c_list])
    print('new_clu: ', len(new_clu))

    all_clu = new_clu

    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    pickle.dump({'cluster': cid_tid_label}, open(cluster_path, 'wb'))
