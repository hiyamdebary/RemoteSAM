#!/bin/bash

FILE="./result/DET/DET.log"
: > $FILE

CATEGORY=("airplane" "airport" "baseballfield" "basketballcourt" "bridge" "chimney" "dam" "expressway-service-area" "expressway-toll-station" "golffield" "groundtrackfield" "harbor" "overpass" "ship" "stadium" "storagetank" "tenniscourt" "trainstation" "vehicle" "windmill")

for i in "${!CATEGORY[@]}"; do
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
        ./tasks/code/eval/DET.py \
        --resume ./pretrained_weights/checkpoint.pth \
        --split test \
        --window12 \
        --save_path "./result/DET/" \
        --task "DET" \
        --dataset "DIOR" \
        --imageFolder "./refer/data/DIORcoco/JPEGImages" \
        --annoFolder "./refer/data/DIORcoco/Annotations/DIOR_test.json" \
        --_class "${CATEGORY[$i]}" \
        --EPOC \
        2>&1 | tee -a $FILE
done