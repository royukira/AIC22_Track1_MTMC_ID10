#JOB_NAME="vit_base_multicamvision_mcam_pretrained"
JOB_NAME="vit_small_multicamvision"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
TOTAL_NODES=1 NODE_ID=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29510 \
./run_tracklet_ibot_eval.sh cityflow_metric $JOB_NAME rit_small teacher 4 work_dirs/$JOB_NAME/checkpoint.pth \
    --num_patches 128 \
    --batch_size_per_gpu 64 \
    --num_labels 892 \
    --center \
    --circle \
    --smooth 0.1 \
    --bottleneck_dim 2048

# ./run_tracklet_ibot_eval.sh cityflow_linear $JOB_NAME rit_small teacher 4 work_dirs/$JOB_NAME/checkpoint.pth \
#     --num_patches 128 \
#     --batch_size_per_gpu 32 \
#     --num_labels 892

# ./run_tracklet_ibot_eval.sh cityflow_knn $JOB_NAME rit_small_v2 teacher 4 work_dirs/$JOB_NAME/checkpoint0760.pth \
#     --num_patches 128 \
#     --batch_size_per_gpu 128

# ./run_tracklet_ibot_eval.sh cityflow_knn $JOB_NAME rit_base teacher 4 work_dirs/$JOB_NAME/checkpoint.pth \
#     --num_patches 128 \
#     --batch_size_per_gpu 128

# ./run_tracklet_ibot_eval.sh cityflow_linear $JOB_NAME rit_small_v2 teacher 4 work_dirs/$JOB_NAME/checkpoint.pth \
#     --num_patches 128 \
#     --batch_size_per_gpu 64 \
#     --num_labels 666

# ./run_tracklet_ibot_eval.sh cityflow_reg $JOB_NAME rit_small_v2 teacher 4 work_dirs/$JOB_NAME/checkpoint.pth \
#     --num_patches 128 \
#     --batch_size_per_gpu 128

# ./run_tracklet_ibot_eval.sh cityflow_unsup_cls $JOB_NAME rit_small_v2 teacher 4 work_dirs/$JOB_NAME/checkpoint.pth \
#     --num_patches 128 \
#     --batch_size_per_gpu 64






