#!/bin/bash

FILE="./result/DET_DOTA/DET_DOTA.log"
: > $FILE

CATEGORY=("airplane" "ship" "storagetank" "baseballfield" "tenniscourt" "basketballcourt" "groundtrackfield" "harbor" "bridge" "large-vehicle" "small-vehicle" "helicopter" "roundabout" "soccer-ball-field" "swimming-pool" "container-crane" "airport" "helipad")

for i in "${!CATEGORY[@]}"; do
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
        ./tasks/code/eval/DET_DOTA.py \
        --resume ./pretrained_weights/checkpoint.pth \
        --split test \
        --window12 \
        --save_path "./result/DET_DOTA/" \
        --task "DET" \
        --dataset "DOTAv2" \
        --imageFolder "./refer/data/DOTAv2_patches/test-dev/images" \
        --annoFolder "./refer/data/DOTAv2_patches/test-dev/annfiles" \
        --_class "${CATEGORY[$i]}" \
        2>&1 | tee -a $FILE
done