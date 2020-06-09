#!/usr/bin/env bash

python main.py \
    -m train \
    --pretrained \
    --config configs/quantization/mobilenetv2_imagenet_int8.json \
    --data /data/datasets/imagenet_jpeg/ilsvrc2012/torchvision/ \
    --gpu-id 2
