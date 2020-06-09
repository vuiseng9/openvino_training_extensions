import os

print(os.getcwd())

print("CUDA_HOME => ", os.environ['CUDA_HOME'])


import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.models as models

from nncf.dynamic_graph import patch_torch_operators
from nncf.algo_selector import create_compression_algorithm

patch_torch_operators()

from nncf.config import Config
from nncf.utils import get_all_modules, get_all_modules_by_type, get_state_dict_names_with_modules

import numpy as np
import pandas as pd

cfgfile = "./examples/classification/configs/quantization/vgg11_imagenet_int8.json"
config = Config.from_json(cfgfile)

vgg11 = models.vgg11(pretrained=True)

algo = create_compression_algorithm(vgg11, config)