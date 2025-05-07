#!/bin/bash

python test.py --swin_type base --dataset rrsisd --resume ./pretrained_weights/checkpoint.pth --split val --workers 4 --window12 --img_size 896

