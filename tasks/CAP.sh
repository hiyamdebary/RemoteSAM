#!/bin/bash

FILE="./result/CAP/CAP.log"
: > $FILE

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
    ./tasks/code/eval/CAP.py \
    --resume ./pretrained_weights/checkpoint.pth \
    --window12 \
    --save_path "./result/CAP/" \
    --task "CAP" \
    --dataset "UCM" \
    --imageFolder "./refer/data/UCM_captions/Images" \
    --annoFolder "./refer/data/UCM_captions/dataset.json" \
    2>&1 | tee -a $FILE