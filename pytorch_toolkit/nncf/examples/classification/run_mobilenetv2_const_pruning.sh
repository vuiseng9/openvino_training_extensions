#!/usr/bin/env bash

python main.py \
    -m train \
    --pretrained \
    --config configs/sparsity/mobilenetv2_imagenet_const_sparsity.json \
    --data /data/datasets/imagenet_jpeg/ilsvrc2012/torchvision/
