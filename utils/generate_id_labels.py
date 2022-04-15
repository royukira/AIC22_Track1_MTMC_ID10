import pickle

if __name__ == "__main__":
    # plk_list = [
    #     "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/train/det_feat/feat/merge_feat_seq/S01_merge_feat_seq.pkl",
    #     "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/validation/det_feat/feat/merge_feat_seq/S02_merge_feat_seq.pkl",
    #     "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/train/det_feat/feat/merge_feat_seq/S03_merge_feat_seq.pkl",
    #     "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/train/det_feat/feat/merge_feat_seq/S04_merge_feat_seq.pkl",
    #     "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/validation/det_feat/feat/merge_feat_seq/S05_merge_feat_seq.pkl"
    # ]

    plk_list = [
        "/home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/train/det_feat/feat/merge_feat_seq/S01_merge_feat_seq.pkl",
    ]

    data_list = []
    for plk in plk_list:
        plk_io = open(plk, 'rb')
        data_list.append(
            pickle.load(plk_io)
        )
        plk_io.close()
    
    data_keys = []
    for data in data_list:
        data_keys += list(data.keys())
    
    print(len(data_keys))