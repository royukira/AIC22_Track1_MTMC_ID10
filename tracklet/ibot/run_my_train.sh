# CUDA_VISIBLE_DEVICES=0,1,2,3 \
TOTAL_NODES=1 NODE_ID=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29510 \
./run_tracklet_ibot.sh cityflow_pretrain vit_small_rev_v2_notau rit_small_v2 teacher 8 \
    --data_path /home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/all_train/ \
    --num_patches 128 \
    --transform_type reverse \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 30 \
    --norm_last_layer false \
    --epochs 800 \
    --batch_size_per_gpu 64 \
    --out_dim 8192 \
    --shared_head true \
    --pred_ratio 0 0.3 \
    --pred_ratio_var 0 0.2 \
    --num_workers 8
    
# ./run_tracklet_ibot.sh cityflow_pretrain vit_base_ibotargs_multicamvision_mcam rit_base teacher 8 \
#     --data_path /home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/all_train/ \
#     --num_patches 128 \
#     --embed_type multicam \
#     --lr 0.00075 \
#     --min_lr 2e-6 \
#     --teacher_temp 0.07 \
#     --warmup_teacher_temp_epochs 50 \
#     --norm_last_layer false \
#     --epochs 800 \
#     --freeze_last_layer 3 \
#     --batch_size_per_gpu 64 \
#     --shared_head true \
#     --out_dim 8192 \
#     --pred_ratio 0 0.7 \
#     --pred_ratio_var 0 0.05 \
#     --num_workers 10 \

# ./run_tracklet_ibot.sh cityflow_pretrain vit_base_multicamvision_mcam_pretrained rit_base teacher 4 \
#     --data_path /home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/all_train/ \
#     --num_patches 128 \
#     --transform_type multicam \
#     --teacher_temp 0.07 \
#     --warmup_teacher_temp_epochs 30 \
#     --norm_last_layer false \
#     --epochs 800 \
#     --batch_size_per_gpu 128 \
#     --shared_head true \
#     --out_dim 8192 \
#     --pred_ratio 0 0.3 \
#     --pred_ratio_var 0 0.2 \
#     --num_workers 10 \
#     --pretrained_weights /home/zhangrui/AIC21-MTMC/models/ibot/vit_b_16_rand.pth \
