#!/bin/bash
FILE="./result/SEG/SEG.log"
: > $FILE


python ./tasks/code/eval/SEG.py --nproc_per_node 1 --master_port 12345 \ 
    --resume ./pretrained_weights/checkpoint.pth \
    --split val \
     --workers 4 \
     --window12 \
     --img_size 896 \
     --save_path "./result/SEG/" \
    --task "SEG" \
    2>&1 | tee -a $FILE

