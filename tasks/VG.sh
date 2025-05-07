#!/bin/bash

FILE="./result/VG/VG.log"
: > $FILE

python \
    ./tasks/code/eval/VG.py \
    --resume ./pretrained_weights/checkpoint.pth \
    --split test \
    --window12 \
    --task "VG" \
    --dataset "RSVG" \
    --imageFolder "./refer/data/RSVG/JPEGImages" \
    --annoFolder "./refer/data/RSVG/Annotations" \
    2>&1 | tee -a $FILE