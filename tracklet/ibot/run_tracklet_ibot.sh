#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
export PYTHONPATH="$PYTHONPATH:$CURDIR"
echo 'The work dir is: ' $CURDIR

TYPE=$1
JOB_NAME=$2
ARCH=$3
KEY=$4
GPUS=$5

if [ $GPUS -lt ${MAX_GPUS:-8} ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-${MAX_GPUS:-8}}
fi

if [ -z $OUTPUT_DIR ];then
    OUTPUT_DIR=$CURDIR/work_dirs/$JOB_NAME
fi

if [ -z $PRETRAINED ];then
    PRETRAINED=$OUTPUT_DIR/checkpoint.pth
fi

# pre-training
if [[ $TYPE =~ pretrain ]]; then
    echo "==> Starting pretrainin iBOT."
    python3 -m torch.distributed.launch --nnodes ${TOTAL_NODES:-1} \
        --node_rank ${NODE_ID:-0} --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-29500} \
        $CURDIR/main_tracklet_ibot.py \
        --arch $ARCH \
        --output_dir $OUTPUT_DIR \
        ${@:6}
fi