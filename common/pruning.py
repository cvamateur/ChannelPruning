import re
import torch
import torch.nn as nn

from typing import Union, Dict, Optional


class Conv2dPrune(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones([self.out_channels], dtype=torch.float32))
        self.sorted_indices = []

    def forward(self, input):
        weight = self.mask.view(-1, 1, 1, 1) * self.weight
        bias = self.mask * self.bias
        return self._conv_forward(input, weight, bias)

    @classmethod
    def from_float(cls, conv):
        assert isinstance(conv, nn.Conv2d)
        new_conv = cls(
            conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
            dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None)
        device = conv.weight.device
        new_conv.to(device)
        new_conv.weight = conv.weight
        new_conv.bias = conv.bias
        return new_conv


def is_dw_conv(mod):
    if isinstance(mod, nn.Conv2d) and mod.groups != 1:
        return True
    return False


def parse_name(name: str, pattern=r"(.+?)(?:\[(\d*)/(\d*)\]|$)"):
    matches = re.match(pattern, name)
    module_name, split_idx, num_splits = matches.groups()
    split_idx = int(split_idx) if split_idx is not None else 1
    num_splits = int(num_splits) if num_splits is not None else 1
    return module_name, split_idx, num_splits


def prepare_channel_pruning(model: nn.Module) -> nn.Module:
    def _swap_module(module: nn.Module):

        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                new_conv = Conv2dPrune.from_float(child)
                module._modules[name] = new_conv
            else:
                _swap_module(child)
        return module

    return _swap_module(model)


def get_structure_info():
    return [
        ["conv0", "conv1"],
        ["conv1", ["branch0.0", "branch1.0"]],
        ["branch0.0", "branch0.2.dw"],
        ["branch0.2.dw", "branch0.2.pw"],
        ["branch0.2.pw", "branch0.4"],
        ["branch0.4", "conv2.dw[1/2]"],
        ["branch1.0", "branch1.2.dw"],
        ["branch1.2.dw", "branch1.2.pw"],
        ["branch1.2.pw", "branch1.4"],
        ["branch1.4", "conv2.dw[2/2]"],
        ["conv2.dw", "conv2.pw"],
        ["conv2.pw", "out"]
    ]


def gradient_based_importance(mod: nn.Module):
    assert hasattr(mod, "weight")
    weight = mod.weight
    if weight.grad is None:
        raise RuntimeError("no gradient is available, backward your model first")
    importance_tensor = (weight.grad * weight).abs()
    if isinstance(mod, nn.Conv2d):
        importance = importance_tensor.sum((1, 2, 3))
    elif isinstance(mod, nn.ConvTranspose2d):
        importance = importance_tensor.sum((0, 2, 3))
    else:
        raise NotImplementedError(type(mod))
    return importance


def get_in_out_channel_dim(mod: nn.Module):
    if isinstance(mod, nn.Conv2d):
        dim_in, dim_out = 1, 0
    elif isinstance(mod, nn.ConvTranspose2d):
        dim_in, dim_out = 0, 1
    else:
        raise RuntimeError(f"module must be nn.Conv2d or nn.ConvTranspose2d, got {type(mod)}")
    return dim_in, dim_out


def sort_params(weight: torch.Tensor, bias: Optional[torch.Tensor], dim: int, indices):
    sorted_weight = torch.index_select(weight, dim, indices)
    weight.copy_(sorted_weight)
    if bias is not None:
        sorted_bias = torch.index_select(bias, 0, indices)
        bias.copy_(sorted_bias)


class ChannelPruner:

    def __init__(self, importance_criterion: callable, pruning_ratio: Union[float, Dict[str, float]]):
        self.criterion = importance_criterion
        self.pruning_ratio = pruning_ratio

    @torch.no_grad()
    def prune(self, model: nn.Module, struct_info: list):
        prepare_channel_pruning(model)

        for parent, children in struct_info:
            parent_mod = model.get_submodule(parent)

            # determine how many channels to keep
            ratio = self.pruning_ratio
            if isinstance(ratio, dict):
                ratio = ratio.get(parent, ratio["default"])
            ratio = float(ratio)
            assert 0 <= ratio < 1
            num_keep_channels = int(parent_mod.out_channels * (1 - ratio))

            # ------------ prune parent node ------------
            # for dw conv, it's sorting index are determined by it parent, when its
            # parent are sorted, the dw conv itself has already attached with an
            # sorted_index attribute
            if is_dw_conv(parent_mod) and parent_mod.sorted_indices:
                indices = parent_mod.sorted_indices
                indices = torch.cat(indices)
                parent_mod.sorted_indices.clear()
            else:
                importance = self.criterion(parent_mod)
                indices = torch.argsort(importance, descending=True)

                # sort parent node output channels
                _, dim_out = get_in_out_channel_dim(parent_mod)
                sort_params(parent_mod.weight.data, parent_mod.bias.data, dim_out, indices)

                # update parent node's mask
                parent_mod.mask[num_keep_channels:] = 0

            # ------------ prune children nodes -------------
            if isinstance(children, str):
                children = [children]

            for child in children:
                child, split_idx, num_splits = parse_name(child)
                child_mod = model.get_submodule(child)


                # the child may be located behind a concat module, the sorting procedure
                # must be taken on the corresponding portion of the weights
                start = parent_mod.out_channels * (split_idx - 1)
                end = parent_mod.out_channels * split_idx
                dim_in, dim_out = get_in_out_channel_dim(child_mod)
                if is_dw_conv(child_mod):
                    dim_to_sort = dim_out
                else:
                    dim_to_sort = dim_in

                child_weight = child_mod.weight.data.narrow(dim_to_sort, start, end-start)
                sort_params(child_weight, None, dim_to_sort, indices)

                # save sorting index
                indices.add_(start)
                child_mod.sorted_indices.append(indices)

        return model
