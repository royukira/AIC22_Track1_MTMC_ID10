#!/bin/bash
### Please move the test dataset(AIC22_Track1_MTMC_Tracking) to './datasets' before running this shell.

MCMT_CONFIG_FILE="temp_aic_all.yml"
REID1_CONFIG_FILE="temp_reid1.yml"
REID2_CONFIG_FILE="temp_reid2.yml"
REID3_CONFIG_FILE="temp_reid3.yml"
cd config/
python process_yml.py "aic_all.yml" ${MCMT_CONFIG_FILE}
python process_yml.py "aic_reid1.yml" ${REID1_CONFIG_FILE}
python process_yml.py "aic_reid2.yml" ${REID2_CONFIG_FILE}
python process_yml.py "aic_reid3.yml" ${REID3_CONFIG_FILE}

### Pre-pocesssing ####
cd ../detector/
python gen_images_aic.py ${MCMT_CONFIG_FILE}

# #### Run Detector.####
cd yolov5/
bash gen_det.sh ${MCMT_CONFIG_FILE}

# #### Extract reid feautres.####
cd ../../reid/
python extract_image_feat.py ${REID1_CONFIG_FILE}
python extract_image_feat.py ${REID2_CONFIG_FILE}
python extract_image_feat.py ${REID3_CONFIG_FILE}
python merge_reid_feat.py ${MCMT_CONFIG_FILE}

# # # # #### ByteTrack with occlusion handling. ####
cd ../tracker/ByteTrack
bash run_aic.sh ${MCMT_CONFIG_FILE}
wait

# # # # # #### Get results. ####
cd ../../reid/reid_matching/tools

# ### === Use Transformer for MC Association (Not be used in the end)  ===========
# ### TODO: Generative MCA instead of similarity matrix?
# # WORK_NAME="vit_small_multicamvision"
# # PRETRAINED="/home/zhangrui/AIC21-MTMC/tracklet/ibot/work_dirs/$WORK_NAME/checkpoint.pth"
# # LCB_PRETRAINED="/home/zhangrui/AIC21-MTMC/tracklet/ibot/work_dirs/$WORK_NAME/eval/metric/checkpoint_teacher_metric.pth"
# # AVGPOOL=2
# # KEY='teacher'
# # ARCH='rit_small'
# # NUM_PATCHES=128
# # NUM_LABELS=892  # S01-S05: 666; S06: 582
# # KEY_LIST=($(echo $KEY | tr "," "\n"))
# # NUM_BOTTLENECK_DIM=2048
# # ====================================================

# ### === Use Avg ReID Features for MC Association =======
WORK_NAME="none"
PRETRAINED="none"
LCB_PRETRAINED="none"
AVGPOOL=0
KEY='none'
ARCH='none'
NUM_PATCHES=0
NUM_LABELS=0
KEY_LIST=($(echo $KEY | tr "," "\n"))
NUM_BOTTLENECK_DIM=0
# # ====================================================


python trajectory_fusion.py \
    --mcmt_config $MCMT_CONFIG_FILE \
    --pretrained_weights $PRETRAINED \
    --linear_cls_weights $LCB_PRETRAINED \
    --avgpool_patchtokens $AVGPOOL \
    --arch $ARCH \
    --checkpoint_key ${KEY_LIST[$K]} \
    --num_patches $NUM_PATCHES \
    --bottleneck_dim $NUM_BOTTLENECK_DIM \
    --num_labels $NUM_LABELS \
    ${@:4}

python sub_cluster.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}

#### Post-Processing ####
python find_outlier_tracklet.py ${MCMT_CONFIG_FILE}

#### Delete temporal configs
cd ../../../config
rm ${MCMT_CONFIG_FILE} ${REID1_CONFIG_FILE} ${REID2_CONFIG_FILE} ${REID3_CONFIG_FILE}