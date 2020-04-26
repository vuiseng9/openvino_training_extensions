import logging
import os.path as osp
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import CIFAR10, CIFAR100

from examples.common.argparser import get_common_argument_parser
from examples.common.model_loader import load_model
from examples.common.distributed import configure_distributed, is_main_process
from examples.common.execution import ExecutionMode, get_device, get_execution_mode, \
    prepare_model_for_execution, start_worker

from nncf.helpers import create_compressed_model, load_state, safe_thread_call
from nncf.dynamic_graph.graph_builder import create_input_infos
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot, \
    print_args, make_additional_checkpoints, get_name, is_binarization
from nncf.config import Config
from nncf.dynamic_graph import patch_torch_operators
from nncf.utils import manual_seed, print_statistics
from examples.common.utils import write_metrics
patch_torch_operators()

from copy import deepcopy
import pandas as pd
import numpy as np
from collections import OrderedDict
import math

# logging
def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

class QuantizationEnv:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config['model']
        
        autoq_cfg = config.get('auto_quantization', {})
        self.min_bit = autoq_cfg['min_bit'] if 'min_bit' in autoq_cfg else 2
        self.max_bit = autoq_cfg['max_bit'] if 'max_bit' in autoq_cfg else 8
        self.compress_ratio = autoq_cfg['compress_ratio'] if 'compress_ratio' in autoq_cfg else 0.125
        self.float_bit = autoq_cfg['float_bit'] if 'float_bit' in autoq_cfg else 32.0

        self.last_action = self.max_bit

        pretrained_model = load_model(
                                model=self.model_name,
                                pretrained=True,
                                num_classes=config.get('num_classes', 1000))

        self.pretrained_model, _ = prepare_model_for_execution(pretrained_model, config)
  
        if config.distributed:
            self.compression_algo.distributed()

        self.is_inception = 'inception' in self.model_name

        # define loss function (criterion)
        self.criterion = nn.CrossEntropyLoss().to(config.device)

        # Data loading code
        self.train_loader, self.train_sampler, self.val_loader = self._create_dataloaders()

        self.state_emb = OrderedDict()

        # init reward
        self.best_reward = -math.inf
        self.evaluate_pretrained_accuracy()
        self.create_compression_algo(self.config)
        self.reset()

    def reset(self):
        self.build_state_embedding()
        self._build_index()
        self._get_weight_size()
        self.cur_ind = 0
        self.strategy = []  # quantization strategy

    def create_compression_algo(self, config):
        self.model = deepcopy(self.pretrained_model)
        self.compression_algo, self.model = create_compressed_model(self.model, config)
        self.model, _ = prepare_model_for_execution(self.model, config)

    def apply_actions(self, strategy):
        for i,(k, v) in enumerate(self.model.all_quantizations.items()):
            if v.is_weights:
                v.set_precision(strategy.pop(0))
            
    def _get_weight_size(self):
        # get the param size for each layers to prune, size expressed in number of params
        self.wsize_list = []
        for i, (mkey, unnormalized) in enumerate(self.state_emb.items()):
            if i in self.quantizable_idx:
                self.wsize_list.append(
                    self.model.quantized_weight_modules[mkey].weight.data.numel()
                    )
        self.wsize_dict = {i: s for i, s in zip(self.quantizable_idx, self.wsize_list)}

    def _cur_weight(self):
        cur_weight = 0.
        # quantized
        for i, n_bit in enumerate(self.strategy):
            cur_weight += n_bit * self.wsize_list[i]
        return cur_weight

    def _org_weight(self):
        org_weight = 0.
        org_weight += sum(self.wsize_list) * self.float_bit
        return org_weight

    def _build_index(self):
        # This method assume build_state_embedding has been performed
        self.quantizable_idx = []
        self.layer_type_list = []
        self.bound_list = []
        for i, (mkey, unnormalized) in enumerate(self.state_emb.items()):
            self.quantizable_idx.append(i)
            self.layer_type_list.append(type(self.model.quantized_weight_modules[mkey]))
            self.bound_list.append((self.min_bit, self.max_bit))
        print('=> Final bound list: {}'.format(self.bound_list))

    def build_state_embedding(self):
        annotate_model_attr(self.model, (3, 224, 224), repl=False)

        self.state_emb.clear()

        for i, (mkey, m) in enumerate(self.model.quantized_weight_modules.items()):
            # print("\n{} ------------------\n{}".format(i, mkey))
            # print(m.__class__.__name__)
            if m.__class__.__name__ == 'NNCFConv2d':
                state_list=[]
                state_list.append(i) # index
                state_list.append(4.0) # qbit
                state_list.append([1.]) # layer type, 1 for conv_dw
                state_list.append(m.in_channels) # in channels
                state_list.append(m.out_channels) # out channels
                state_list.append(m.stride[0]) # stride
                state_list.append(m.kernel_size[0]) # kernel size
                state_list.append(np.prod(m.weight.size())) # weight size
                state_list.append(np.prod(m._input_shape[-2:])) # input feature_map_size
                self.state_emb[mkey]=np.hstack(state_list)
            elif m.__class__.__name__ == 'NNCFLinear':
                state_list=[]
                state_list.append(i) # index
                state_list.append(4.0) # qbit
                state_list.append([0.])  # layer type, 0 for fc
                state_list.append([m.in_features]) # in features
                state_list.append([m.out_features]) # out features
                state_list.append([0.])  # stride
                state_list.append([1.])  # kernel size
                state_list.append([np.prod(m.weight.size())])  # weight size
                state_list.append(m._input_shape[-1])
                self.state_emb[mkey]=np.hstack(state_list)

        emb_list = []
        for i, (mkey, v) in enumerate(self.state_emb.items()):
            emb_list.append(v)

        layer_embedding = np.array(emb_list, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)
        self.layer_embedding = layer_embedding
    
    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind
        # limit the action to certain range
        action = float(action)
        min_bit, max_bit = self.bound_list[self.cur_ind]
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        self.last_action = action
        return action  # not constrained here

    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1

    def _final_action_wall(self):
        target = self.compress_ratio * self._org_weight()
        min_weight = 0
        for i, n_bit in enumerate(self.strategy):
            min_weight += self.wsize_list[i] * self.min_bit
        while min_weight < self._cur_weight() and target < self._cur_weight():
            for i, n_bit in enumerate(reversed(self.strategy)):
                if n_bit > self.min_bit:
                    self.strategy[-(i+1)] -= 1
                if target >= self._cur_weight():
                    break
        print('=> Final action list: {}'.format(self.strategy))

    def reward(self, acc, w_size_ratio=None):
        if w_size_ratio is not None:
            return (acc - self.org_acc + 1. / w_size_ratio) * 0.1
        return (acc - self.org_acc) * 0.1

    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        action = self._action_wall(action)  # percentage to preserve

        self.strategy.append(action)  # save action to strategy

        # all the actions are made
        if self._is_final_layer():
            self._final_action_wall()
            assert len(self.strategy) == len(self.quantizable_idx)
            w_size = self._cur_weight()
            w_size_ratio = self._cur_weight() / self._org_weight()

            override_cfg = {}
            for i, layer_id in enumerate(self.state_emb.keys()):
                override_cfg[layer_id] = {"bits": self.strategy[i]}
            self.config.compression['scope_overrides'] = override_cfg

            # Quantization
            self.apply_actions(self.strategy)

            for k, v in self.model.all_quantizations.items():
                print("### Quantization action", k, v)

            self.compression_algo.initialize(self.train_loader)
            _, acc =self._validate(self.model)
            # centroid_label_dict = quantize_model(self.model, self.quantizable_idx, self.strategy,
            #                                      mode='cpu', quantize_bias=False, centroids_init='k-means++',
            #                                      is_pruned=self.is_model_pruned, max_iter=3)

            # if self.finetune_flag:
            #     train_acc = self._kmeans_finetune(self.train_loader, self.model, self.quantizable_idx,
            #                                       centroid_label_dict, epochs=self.finetune_epoch, verbose=False)
            #     acc = self._validate(self.val_loader, self.model)
            # else:
            #     acc = self._validate(self.val_loader, self.model)

            # reward = self.reward(acc, w_size_ratio)
            reward = self.reward(acc)

            info_set = {'w_ratio': w_size_ratio, 'accuracy': acc, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, acc, w_size_ratio))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set

        w_size = self._cur_weight()
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        self.layer_embedding[self.cur_ind][-1] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        return obs, reward, done, info_set

    def evaluate_pretrained_accuracy(self):
        print("Evaluating pretrained accuracy")
        # self.pretrained_top1, self.pretrained_top5 = self._validate(self.pretrained_model)
        self.pretrained_top1 = 69.02	
        self.pretrained_top5 = 88.63
        self.org_acc = self.pretrained_top5

    def _validate(self, model):
        criterion = self.criterion
        config = self.config
        val_loader = self.val_loader

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input_, target) in enumerate(val_loader):
                input_ = input_.to(config.device)
                target = target.to(config.device)

                # compute output
                output = model(input_)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss, input_.size(0))
                top1.update(acc1, input_.size(0))
                top5.update(acc5, input_.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    print(
                        '{rank}'
                        'Test: [{0}/{1}] '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                        'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5,
                            rank='{}:'.format(config.rank) if config.multiprocessing_distributed else ''
                        ))

            if is_main_process():
                config.tb.add_scalar("val/loss", losses.avg, len(val_loader) * config.get('cur_epoch', 0))
                config.tb.add_scalar("val/top1", top1.avg, len(val_loader) * config.get('cur_epoch', 0))
                config.tb.add_scalar("val/top5", top5.avg, len(val_loader) * config.get('cur_epoch', 0))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
            print()

            if config.metrics_dump is not None:
                avg = round(top1.avg, 2)
                if config.resuming_checkpoint is None:
                    model_name = os.path.basename(config.config).replace(".json", ".pth")
                else:
                    model_name = (os.path.basename(config.resuming_checkpoint))
                metrics = {model_name: avg}
                write_metrics(config, metrics)
        return top1.avg, top5.avg

    def _create_dataloaders(self):
        def get_dataset(dataset_config, config, transform, is_train):
            if dataset_config == 'imagenet':
                prefix = 'train' if is_train else 'val'
                return datasets.ImageFolder(osp.join(config.dataset_dir, prefix), transform)
            return create_cifar(config, dataset_config, is_train, transform)

        config = self.config
        dataset_config = config.dataset if config.dataset is not None else 'imagenet'
        dataset_config = dataset_config.lower()
        assert dataset_config in ['imagenet', 'cifar100', 'cifar10'], "Unknown dataset option"

        if dataset_config == 'imagenet':
            normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))
        elif dataset_config == 'cifar100':
            normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                            std=(0.2023, 0.1994, 0.2010))
        elif dataset_config == 'cifar10':
            normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                            std=(0.5, 0.5, 0.5))

        input_info_list = create_input_infos(config)
        image_size = input_info_list[0].shape[-1]
        size = int(image_size / 0.875)
        val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

        val_dataset = get_dataset(dataset_config, config, val_transform, is_train=False)

        pin_memory = config.execution_mode != ExecutionMode.CPU_ONLY

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        batch_size = int(config.batch_size)
        workers = int(config.workers)
        if config.execution_mode == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
            batch_size //= config.ngpus_per_node
            workers //= config.ngpus_per_node

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=pin_memory)

        if config.mode.lower() == "test":
            return None, None, val_loader

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = get_dataset(dataset_config, config, train_transforms, is_train=True)

        if config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=pin_memory, sampler=train_sampler)
        return train_loader, train_sampler, val_loader


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def create_cifar(config, dataset_config, is_train, transform):
    create_cifar_fn = None
    if dataset_config == 'cifar100':
        create_cifar_fn = partial(CIFAR100, config.dataset_dir, train=is_train, transform=transform)
    if dataset_config == 'cifar10':
        create_cifar_fn = partial(CIFAR10, config.dataset_dir, train=is_train, transform=transform)
    if create_cifar_fn:
        return safe_thread_call(partial(create_cifar_fn, download=True), partial(create_cifar_fn, download=False))
    return None

def annotate_model_attr(model, input_size, batch_size=-1, repl=False):
    _ = get_model_attr(model, input_size, batch_size, repl)

def get_model_attr(model, input_size, batch_size=-1, repl=False):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            
            module._input_shape=summary[m_key]["input_shape"]
            module._output_shape=summary[m_key]["output_shape"]
            module._nb_params=summary[m_key]["nb_params"]
            
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

#     device = device.lower()
#     assert device in [
#         "cuda",
#         "cpu",
#     ], "Input device is not valid, please specify 'cuda' or 'cpu'"

#     if device == "cuda" and torch.cuda.is_available():
    if next(model.parameters()).is_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if repl:
        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")

        return summary