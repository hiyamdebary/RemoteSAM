#!/bin/bash

FILE="./result/MLC/MLC.log"
: > $FILE

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
    ./tasks/code/eval/MLC.py \
    --resume ./pretrained_weights/checkpoint.pth \
    --split test \
    --window12 \
    --save_path "./result/MLC/" \
    --task "MLC" \
    --dataset "DIOR" \
    --imageFolder "./refer/data/DIORcoco/JPEGImages" \
    --annoFolder "./refer/data/DIORcoco/Annotations/DIOR_test.json" \
    2>&1 | tee -a $FILE