#!/bin/bash

FILE="./result/CNT/CNT.log"
: > $FILE

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
    ./tasks/code/eval/CNT.py \
    --resume ./pretrained_weights/checkpoint.pth \
    --split test \
    --window12 \
    --save_path "./result/CNT/" \
    --task "CNT" \
    --dataset "DIOR" \
    --imageFolder "./refer/data/Counting/test/DIOR/Images" \
    --annoFolder "./refer/data/Counting/test/DIOR/Annotation/test_IMG_CT.json" \
    --EPOC \
    2>&1 | tee -a $FILE