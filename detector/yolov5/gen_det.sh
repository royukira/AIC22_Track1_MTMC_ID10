#!/bin/bash
seqs=(c041 c042 c043 c044 c045 c046)   # test/s06

gpu_id=0
gpu_last_id=3
for seq in ${seqs[@]}
do
    echo "$seq -- Using GPU:$gpu_id"
    CUDA_VISIBLE_DEVICES=${gpu_id} python detect2img.py \
        --name ${seq} \
        --weights ../../models/detector/yolov5/yolov5x.pt \
        --conf 0.1 \
        --agnostic \
        --save-txt \
        --save-conf \
        --img-size 1280 \
        --classes 2 5 7 \
        --cfg_file $1&
    
    if [[ $gpu_id = $gpu_last_id ]]; then
        gpu_id=0
    else
        gpu_id=$(($gpu_id+1))
    fi

done
wait
