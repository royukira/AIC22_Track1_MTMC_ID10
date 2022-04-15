#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
export PYTHONPATH="$PYTHONPATH:$CURDIR"
echo 'The work dir is: ' $CURDIR
echo 'IP Adress: ' $MASTER_ADDR
echo 'Port: ' $MASTER_PORT

TYPE=$1
JOB_NAME=$2
ARCH=$3
KEY=$4
GPUS=$5
PRETRAINED=$6


if [ $GPUS -lt ${MAX_GPUS:-8} ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-${MAX_GPUS:-8}}
fi

if [ -z $OUTPUT_DIR ];then
    OUTPUT_DIR=$CURDIR/work_dirs/$JOB_NAME/eval/
fi

#evaluation
if [[ $TYPE =~ cityflow_knn ]] || [[ $TYPE =~ cityflow_reg ]] || \
   [[ $TYPE =~ cityflow_linear ]] || [[ $TYPE =~ cityflow_cls ]] || \
   [[ $TYPE =~ cityflow_semi_cls ]] || [[ $TYPE =~ cityflow_unsup_cls ]] ||\
   [[ $TYPE =~ cityflow_metric ]]; then
    
    if [ -z $AVGPOOL ] && [ -z $LINEAR_AVGPOOL ] && [ -z $LINEAR_N_LAST_BLOCKS ]; then
        if [[ $ARCH =~ small ]] || [[ $ARCH =~ swin ]]; then
            AVGPOOL=0
            LINEAR_AVGPOOL=0
            LINEAR_N_LAST_BLOCKS=4
        else
            AVGPOOL=0
            LINEAR_AVGPOOL=2
            LINEAR_N_LAST_BLOCKS=1
        fi
    fi

    echo "==> Starting evaluating iBOT."
    KEY_LIST=($(echo $KEY | tr "," "\n"))

    if [[ $TYPE =~ cityflow_knn ]]; then
        if [[ $TYPE =~ cityflow_knn ]] && [[ ! $TYPE =~ pretrain ]]; then
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_knn.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path eval_dataset/ \
                    --dump_features $OUTPUT_DIR \
                    ${@:7}
            done
        else
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-${K}] \
                    $CURDIR/evaluation/eval_knn.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path eval_dataset/ \
                    --dump_features $OUTPUT_DIR
            done
        fi
    fi
    if [[ $TYPE =~ cityflow_reg ]]; then
        if [[ $TYPE =~ cityflow_reg ]] && [[ ! $TYPE =~ pretrain ]]; then
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_logistic_regression.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path eval_dataset/ \
                    --dump_features $OUTPUT_DIR \
                    ${@:7}
            done
        else
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_logistic_regression.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path eval_dataset/ \
                    --dump_features $OUTPUT_DIR
            done
        fi
    fi
    if [[ $TYPE =~ cityflow_linear ]]; then    
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do  
            if [[ $TYPE =~ cityflow_linear_solo ]] && [[ ! $TYPE =~ pretrain ]]; then
                SUB_OUTPUT_DIR=$OUTPUT_DIR/linear_solo
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${METIS_WORKER_0_PORT:-29500}-$K] \
                    ${CURDIR}/evaluation/eval_linear.py \
                    --pretrained_weights $PRETRAINED \
                    --n_last_blocks $LINEAR_N_LAST_BLOCKS \
                    --avgpool_patchtokens $LINEAR_AVGPOOL \
                    --arch ${ARCH} \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --output_dir $SUB_OUTPUT_DIR \
                    --data_path eval_dataset/ \
                    ${@:7}
            elif [[ $TYPE =~ cityflow_linear ]] && [[ ! $TYPE =~ pretrain ]]; then
                SUB_OUTPUT_DIR=$OUTPUT_DIR/linear
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_linear_multi.py \
                    --pretrained_weights $PRETRAINED \
                    --n_last_blocks $LINEAR_N_LAST_BLOCKS \
                    --avgpool_patchtokens $LINEAR_AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --output_dir $SUB_OUTPUT_DIR \
                    --data_path eval_dataset/ \
                    ${@:7}
            else
                SUB_OUTPUT_DIR=$OUTPUT_DIR/linear
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_linear_multi.py \
                    --pretrained_weights $PRETRAINED \
                    --n_last_blocks $LINEAR_N_LAST_BLOCKS \
                    --avgpool_patchtokens $LINEAR_AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --output_dir $SUB_OUTPUT_DIR \
                    --data_path data/imagenet
            fi
        done
    fi
    if [[ $TYPE =~ cityflow_metric ]]; then    
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do  
            SUB_OUTPUT_DIR=$OUTPUT_DIR/metric
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/eval_metric_learning.py \
                --pretrained_weights $PRETRAINED \
                --n_last_blocks $LINEAR_N_LAST_BLOCKS \
                --avgpool_patchtokens 2 \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --data_path eval_dataset/ \
                ${@:7}
        done
    fi
    if [[ $TYPE =~ cityflow_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/imnet
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            WEIGHT_FILE=$SUB_OUTPUT_DIR/checkpoint_${KEY_LIST[$K]}.pth
            python3 $CURDIR/evaluation/classification_layer_decay/extract_backbone_weights.py \
                $PRETRAINED $WEIGHT_FILE --checkpoint_key ${KEY_LIST[$K]}
            python3 -m torch.distributed.launch --nnodes ${TOTAL_NODES:-1} \
                --node_rank ${NODE_ID:-0} --nproc_per_node=$GPUS_PER_NODE \
                --master_addr=${MASTER_ADDR:-127.0.0.1} \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/classification_layer_decay/run_class_finetuning.py \
                --finetune $WEIGHT_FILE \
                --model $ARCH \
                --epochs 100 \
                --warmup_epochs 20 \
                --layer_decay 0.65 \
                --mixup 0.8 \
                --cutmix 1.0 \
                --layer_scale_init_value 0.0 \
                --disable_rel_pos_bias \
                --abs_pos_emb \
                --use_cls \
                --imagenet_default_mean_and_std \
                --output_dir $SUB_OUTPUT_DIR \
                --data_path data/imagenet \
                ${@:7}
        done
    fi
    if [[ $TYPE =~ cityflow_semi_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=${OUTPUT_DIR}/semi_cls
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/semi_supervised/eval_cls.py \
                --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --data_path data/imagenet_split \
                ${@:7}
        done
    fi
    if [[ $TYPE =~ cityflow_unsup_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do        
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/unsupervised/unsup_cls.py \
                --pretrained_weights $PRETRAINED \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --data_path /home/zhangrui/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/all_train \
                ${@:7}
        done
    fi
fi