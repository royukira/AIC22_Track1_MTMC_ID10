import numpy as np
import pickle
import os

scene_name = 'S06'
plk = open(f"/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/test_baseline/train_data/feat/merge_feat/{scene_name}_merge_feat.pkl",'rb')
data = pickle.load(plk)
plk.close()

new_data = {}
for k, v in data.items():
    feat_seq = []
    for i in range(len(v)):
        feat_seq.append(v[i]['feature'])
    feat_seq_np = np.array(feat_seq)
    #print(feat_seq_np.shape)
    if k in new_data:
        raise AssertionError("Duplicate key.")
    new_data.setdefault(
        k, 
        {
            "seq_feat":feat_seq_np,
            "seq_info":v
        }
    )

new_data_dir = f"/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/test_baseline/train_data/feat/merge_feat_seq/"
if os.path.exists(new_data_dir) is False:
    os.makedirs(new_data_dir)
new_data_path = os.path.join(new_data_dir, f"{scene_name}_merge_feat_seq.pkl")

pickle.dump(new_data, open(new_data_path, 'wb'), pickle.HIGHEST_PROTOCOL)