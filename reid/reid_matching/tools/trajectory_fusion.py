import os
from os.path import join as opj
import numpy as np
import pickle
from utils.zone_intra import zone
import sys
sys.path.append('../../../')
from config import cfg

sys.path.append('../../../tracklet/ibot/')
sys.path.append('../../../tracklet/ibot/evaluation/')
from tracklet.ibot import models as tmodels
from tracklet.ibot import utils as tutils
from tracklet.ibot.evaluation.eval_metric_learning import LinearClassBlock as LCB

import argparse
import torch


def parse_pt(pt_file,zones):
    if not os.path.isfile(pt_file):
        return dict()
    with open(pt_file,'rb') as f:
        lines = pickle.load(f)
    mot_list = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:])
        tid = lines[line]['id']
        bbox = list(map(lambda x:int(float(x)), lines[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = lines[line]
        out_dict['zone'] = zones.get_zone(bbox)
        mot_list[tid][fid] = out_dict
    return mot_list

def parse_bias(timestamp_dir, scene_name):
    cid_bias = dict()
    for sname in scene_name:
        with open(opj(timestamp_dir, sname + '.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                cid = int(line[0][2:])
                bias = float(line[1])
                if cid not in cid_bias: cid_bias[cid] = bias
    return cid_bias

def out_new_mot(mot_list,mot_path):
    out_dict = dict()
    for tracklet in mot_list:
        tracklet = mot_list[tracklet]
        for f in tracklet:
            out_dict[tracklet[f]['imgname']]=tracklet[f]
    pickle.dump(out_dict,open(mot_path,'wb'))


def resize_feat_len(feat:np.ndarray, num_patches):
        if feat.shape[0] == num_patches:
            return feat
        
        # 裁剪头尾段
        if feat.shape[0] > num_patches:
            start_offset = num_patches // 2
            end_offset = feat.shape[0] - num_patches // 2
            if num_patches % 2 != 0:
                new_feat = np.vstack(
                    (feat[:start_offset+1,:], 
                    feat[end_offset:,:])
                )
            else:
                new_feat = np.vstack(
                    (feat[:start_offset,:], 
                    feat[end_offset:,:])
                )
            return new_feat

        # Padding 0
        if feat.shape[0] < num_patches:
            pad_zero = np.zeros((num_patches-feat.shape[0], feat.shape[1]))
            new_feat = np.vstack((feat, pad_zero))
            return new_feat


@torch.no_grad()
def extract_features_single(model, linear_cls_block, tracklet_features:np.ndarray, n_last_blocks, avgpool, num_patches):
    resized_feat = resize_feat_len(tracklet_features.copy(), num_patches)
    input_feat = torch.from_numpy(resized_feat).unsqueeze(0)
    input_feat = input_feat.cuda(non_blocking=True)
    intermediate_output = model.get_intermediate_layers(input_feat, n_last_blocks)
    if avgpool == 0:
        #print("USE CLS")
        # norm(x[:, 0]) CLS
        output = [x[:, 0] for x in intermediate_output]
    elif avgpool == 1:
        # x[:, 1:].mean(1)
        #print("USE PATCH")
        output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
    elif avgpool == 2:
        # norm(x[:, 0]) + norm(x[:, 1:]).mean(1)
        # print("USE CLS+PATCH")
        # print(torch.mean(intermediate_output[-1][:, 1:], dim=1).shape)
        output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
    else:
        assert False, "Unkown avgpool type {}".format(avgpool)
    
    # feats = torch.cat(output, dim=-1).clone().squeeze()
    feats = torch.cat(output, dim=-1)
    _, feats = linear_cls_block(feats)

    return feats.squeeze().cpu().detach().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Trajectory features extractor')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base-size models with [CLS] token when doing linear classification.""")
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument('--linear_cls_weights', default='', type=str, help="""Path to linear classifier block 
        weights to evaluate.""")
    parser.add_argument('--arch', default='none', type=str, choices=[
            'none',
            'rit_small_v2','rit_base_v2',
            'rit_tiny', 'rit_small', 'rit_base', 'rit_large'
            'vit_tiny', 'vit_small', 'vit_base', 'vit_large', 
            'deit_tiny', 'deit_small',
            'swin_tiny','swin_small', 'swin_base', 'swin_large'
        ], help='Architecture.')
    parser.add_argument('--num_patches', default=128, type=int, help='Num of patches.')
    parser.add_argument('--num_labels', default=892, type=int, help='Num of labels.')
    parser.add_argument('--bottleneck_dim', default=1024, type=int, help='')


    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--mcmt_config", default="aic_train.yml", type=str,
        help='Path to mcmt config')
    args = parser.parse_args()
    #tutils.init_distributed_mode(args)

    cfg.merge_from_file(f'../../../config/{args.mcmt_config}')
    cfg.freeze()
    scene_name = ['S06']
    
    data_dir = cfg.DATA_DIR
    det_dir = os.path.join(cfg.DATA_DIR, 'detect_result')
    fea_dir = os.path.join(cfg.DATA_DIR, 'feature_merge')
    mot_dir = os.path.join(cfg.DATA_DIR, 'mot_result')
    out_dir = cfg.OUTPUT_DIR
    save_dir = f'{out_dir}/trajectory/'

    cid_bias = parse_bias(cfg.CID_BIAS_DIR, scene_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cam_paths = os.listdir(mot_dir)
    cam_paths = list(filter(lambda x: 'c' in x, cam_paths))
    cam_paths.sort()
    zones = zone()

    # ============ building network ... ============
    if args.arch in tmodels.__dict__.keys() and 'rit' in args.arch:
        model = tmodels.__dict__[args.arch](
            num_patches=args.num_patches,
            num_classes=0,
            use_mean_pooling=args.avgpool_patchtokens==1
        )
        embed_dim = model.embed_dim
        print(f"Model {args.arch}-{args.num_patches} built.")

        if 'swin' in args.arch:
            num_features = []
            for i, d in enumerate(model.depths):
                num_features += [int(model.embed_dim * 2 ** i)] * d
            feat_dim = sum(num_features[-args.n_last_blocks:])
        else:
            feat_dim = embed_dim * (args.n_last_blocks * int(args.avgpool_patchtokens != 1) + \
                int(args.avgpool_patchtokens > 0))

        linear_cls_block = LCB(feat_dim, class_num=args.num_labels, return_f=True, num_bottleneck=args.bottleneck_dim)

        tutils.load_pretrained_weights(
                model, 
                args.pretrained_weights, 
                args.checkpoint_key, 
                args.arch, 
                is_training=False)
        model.cuda()

        lcb_state_dict = torch.load(args.linear_cls_weights, map_location="cpu")
        best_acc_idx = lcb_state_dict['best_acc_hidx']
        print(f"Best acc idx: {best_acc_idx}")
        best_lcb_state_dict = {".".join(k.split('.')[2:]):v for k, v in lcb_state_dict['state_dict'].items() if int(k.split('.')[0]) == best_acc_idx}   # except 'idx.module'
        linear_cls_block.load_state_dict(best_lcb_state_dict, strict=True)
        linear_cls_block.cuda()
        
        model.eval()
        linear_cls_block.eval()
    else:
        model = None
        linear_cls_block = None

    for cam_path in cam_paths:
        print('processing {}...'.format(cam_path))
        cid = int(cam_path[-3:])
        f_w = open(opj(save_dir, '{}.pkl'.format(cam_path)), 'wb')
        cur_bias = cid_bias[cid]
        mot_path = opj(mot_dir, cam_path,'{}_mot_feat.pkl'.format(cam_path))
        new_mot_path = opj(mot_dir, cam_path, '{}_mot_feat_break.pkl'.format(cam_path))
        print(new_mot_path)
        zones.set_cam(cid)
        mot_list = parse_pt(mot_path,zones)
        mot_list = zones.break_mot(mot_list, cid)
        # mot_list = zones.comb_mot(mot_list, cid)
        mot_list = zones.filter_mot(mot_list, cid) # filter by zone
        mot_list = zones.filter_bbox(mot_list, cid)  # filter bbox
        out_new_mot(mot_list, new_mot_path)

        tid_data = dict()
        for tid in mot_list:
            if cid not in [41,43,46,42,44,45]:
                break
            tracklet = mot_list[tid]
            if len(tracklet) <= 1: continue

            frame_list = list(tracklet.keys())
            frame_list.sort()
            # if tid==11 and cid==44:
            #     print(tid)
            zone_list = [tracklet[f]['zone'] for f in frame_list]
            feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3]-tracklet[f]['bbox'][1])*(tracklet[f]['bbox'][2]-tracklet[f]['bbox'][0])>2000]
            if len(feature_list)<2:
                feature_list = [tracklet[f]['feat'] for f in frame_list]
            io_time = [cur_bias + frame_list[0] / 10., cur_bias + frame_list[-1] / 10.] # (IN, OUT)
            
            all_feat = np.array([feat for feat in feature_list])
            if model is None:
                final_feat = np.mean(all_feat, axis=0)
            else:
                final_feat = extract_features_single(
                    model, 
                    linear_cls_block,
                    all_feat,
                    args.n_last_blocks, 
                    args.avgpool_patchtokens,
                    args.num_patches
                )

            tid_data[tid]={
                'cam': cid,
                'tid': tid,
                'mean_feat': final_feat,
                'zone_list':zone_list,
                'frame_list': frame_list,
                'tracklet': tracklet,
                'io_time': io_time
            }
            
        pickle.dump(tid_data,f_w)
        f_w.close()
