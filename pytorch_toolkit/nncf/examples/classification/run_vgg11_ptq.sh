#!/usr/bin/env bash

python main.py \
    -m ptq \
    --pretrained \
    --config configs/quantization/vgg11_imagenet_int8_ptq.json \
    --data /data/datasets/imagenet_jpeg/ilsvrc2012/torchvision/ \
    --gpu-id 2
