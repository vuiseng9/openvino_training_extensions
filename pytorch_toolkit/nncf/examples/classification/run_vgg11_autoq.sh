#!/usr/bin/env bash

python auto_quantize.py \
    -m autoq \
    --pretrained \
    --config configs/quantization/vgg11_imagenet_autoq.json \
    --data /data/datasets/imagenet_jpeg/ilsvrc2012/torchvision/ \
    --gpu-id 2
