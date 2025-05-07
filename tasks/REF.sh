#!/bin/bash
FILE="./result/REF/REF.log"
: > $FILE


python ./tasks/code/eval/REF.py --nproc_per_node 1 --master_port 12345 \ 
    --swin_type base \
    --dataset rrsisd \
    --resume ./pretrained_weights/checkpoint.pth \
    --split val \
    --workers 4 \
    --window12 \
    --img_size 896 \
    2>&1 | tee -a $FILE

