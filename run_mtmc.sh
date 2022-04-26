#### Single-cam tracking -> Multi-cam trajectories association -> Post-processing
## For saving time, you can download the intermidiate results from Google Drive or Baidu Drive 
## The intermidiate results include detection boxes and ReID features.
## Unzip the downloaded file, and move the directory to './datasets' before running this shell script

MCMT_CONFIG_FILE="temp_aic_all.yml"
cd config/
python process_yml.py "aic_all.yml" ${MCMT_CONFIG_FILE}

cd ../detector/
python gen_images_aic.py ${MCMT_CONFIG_FILE}

# # # # #### ByteTrack with occlusion handling. ####
cd ../tracker/ByteTrack
bash run_aic.sh ${MCMT_CONFIG_FILE}
wait

# # # # # #### Get results. ####
cd ../../reid/reid_matching/tools

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
rm ${MCMT_CONFIG_FILE}