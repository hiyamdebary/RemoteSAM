#!/bin/bash

FILE="./result/MCC/MCC.log"
: > $FILE

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
    ./tasks/code/eval/MCC.py \
    --resume ./pretrained_weights/checkpoint.pth \
    --split test \
    --window12 \
    --save_path "./result/MCC/" \
    --task "MCC" \
    --dataset "UCM" \
    --imageFolder "./refer/data/UCMerced_LandUse/Images" \
    --annoFolder "./refer/data/UCMerced_LandUse/Images" \
    2>&1 | tee -a $FILE