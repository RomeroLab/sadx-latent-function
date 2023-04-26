""" This implementation of mixout is from https://github.com/bloodwass/mixout.
    I made slight modifications for readability. """
from copy import deepcopy
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction


class Mixout(InplaceFunction):
    @staticmethod
    def _make_noise(x: torch.Tensor) -> torch.Tensor:
        return x.new().resize_as_(x)

    @classmethod
    def forward(cls,
                ctx,
                weight: torch.Tensor,
                original_weight: torch.Tensor = None,
                mixout_rate: float = 0.0,
                training: bool = False,
                inplace: bool = False) -> torch.Tensor:

        if mixout_rate < 0 or mixout_rate > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1, but got {mixout_rate}")

        if original_weight is not None and weight.size() != original_weight.size():
            raise ValueError(
                f"A target tensor size must match with a input tensor size {weight.size()}, but got {original_weight.size()}")

        ctx.p = mixout_rate
        ctx.training = training

        if original_weight is None:
            original_weight = cls._make_noise(weight)
            original_weight.fill_(0)
        original_weight = original_weight.to(weight.device)

        if inplace:
            ctx.mark_dirty(weight)
            output = weight
        else:
            output = weight.clone()

        if ctx.p == 0 or not ctx.training:
            return output

        ctx.noise = cls._make_noise(weight)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(weight.size()[0], *([1] * (len(weight.size()) - 1)))
        ctx.noise.expand_as(weight)

        if ctx.p == 1:
            output = original_weight.clone()
        else:
            output = ((1 - ctx.noise) * original_weight + ctx.noise * output - ctx.p * original_weight) / (1 - ctx.p)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Optional[torch.Tensor]:
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(weight: torch.Tensor,
           original_weight: torch.Tensor = None,
           mixout_rate: float = 0.0,
           training: bool = False,
           inplace: bool = False) -> torch.Tensor:

    return Mixout.apply(weight, original_weight, mixout_rate, training, inplace)


class MixLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 original_weight: torch.Tensor = None,
                 mixout_rate: float = 0.0) -> None:

        super(MixLinear, self).__init__(in_features, out_features, bias)

        self.original_weight = original_weight
        self.mixout_rate = mixout_rate

        if self.mixout_rate < 0 or self.mixout_rate > 1:
            raise ValueError(f"Mixout rate should be between 0 and 1, but got {self.mixout_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, mixout(self.weight, self.original_weight,
                                  self.mixout_rate, self.training), self.bias)


def mixout_sub(module: nn.Module, mixout_rate: float, sub_dropout: bool = False) -> nn.Module:
    if isinstance(module, nn.Dropout) and sub_dropout:
        return nn.Dropout(0)
    elif isinstance(module, nn.Linear):
        target_state_dict = deepcopy(module.state_dict())

        bias = True if module.bias is not None else False
        new_module = MixLinear(
            module.in_features,
            module.out_features,
            bias,
            target_state_dict["weight"],
            mixout_rate
        )
        new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module


def recursive_setattr(obj: Any, attr: str, value: Any) -> None:
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


def apply_mixout(model: nn.Module,
                 mixout_rate: float,
                 sub_dropout: bool = False) -> None:

    for name, module in tuple(model.named_modules()):
        if name:
            recursive_setattr(model, name, mixout_sub(module, mixout_rate, sub_dropout))
