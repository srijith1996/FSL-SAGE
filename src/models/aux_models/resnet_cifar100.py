# Define CIFAR ResNet impementations from https://arxiv.org/abs/1512.03385
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from models import aux_models
from typing import Type, Union, List, Optional, Callable, Any

from models.resnet_cifar100 import _weights_init, BasicBlock
from models.aux_models import register_auxiliary_model

# Taken from https://github.com/fcakyon/cifar100-resnet/blob/main/cifarresnet.py
class ResNetAuxiliary(aux_models.GradScalarAuxiliaryModel):
    def __init__(
            self, 
            server,
            block : Type[Union[BasicBlock]], 
            num_blocks:  List[int],
            in_planes : int = 128,
            num_classes : int = 1000,
            device = 'cpu',
            align_epochs=5, 
            align_step=1e-3,
            align_batch_size=100, 
            max_dataset_size=1000
    ) -> None:
        super(ResNetAuxiliary, self).__init__(
            server, device, align_epochs, align_batch_size, max_dataset_size
        )

        self.in_planes = in_planes

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.linear = nn.Linear(32, num_classes)

        self.apply(_weights_init)
        self.align_epochs = align_epochs

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_imp(self, x : Tensor) -> Tensor:
        out = self.layer2(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def forward_inner(self, x : Tensor) -> Tensor:
        return F.log_softmax(self._forward_imp(x), dim =1)


@register_auxiliary_model("resnet56", disable_check=True)
def resnet56_sl_aux(
    server, layers = [9, 9, 9], in_planes : int=512, weights: Optional[Any] =
    None, progress: bool = True, num_classes: int = 100, device='cpu', **kwargs:
    Any
):
    if layers is None: layers = [9, 9, 9]
    return ResNetAuxiliary(
        server, BasicBlock, layers, in_planes, num_classes=num_classes,
        device=device, **kwargs
    )

@register_auxiliary_model("resnet110", disable_check=True)
def resnet110_sl_aux(
    server, layers = [18, 18, 18], in_planes : int=512, weights: Optional[Any] =
    None, progress: bool = True, num_classes: int = 100, device='cpu', **kwargs:
    Any
):
    if layers is None: layers = [18, 18, 18]
    return ResNetAuxiliary(
        server, BasicBlock, layers, in_planes, num_classes=num_classes,
        device=device, **kwargs
    )

@register_auxiliary_model("resnet1202", disable_check=True)
def resnet1202_sl_aux(
    server, layers = [200, 200, 200], in_planes : int=512, weights:
    Optional[Any] = None, progress: bool = True, num_classes: int = 100,
    device='cpu', **kwargs: Any
):
    if layers is None: layers = [200, 200, 200]
    return ResNetAuxiliary(
        server, BasicBlock, layers, in_planes, num_classes=num_classes,
        device=device, **kwargs
    )
