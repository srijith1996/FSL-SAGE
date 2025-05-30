# Define CIFAR ResNet impementations from https://arxiv.org/abs/1512.03385
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

from models import register_client_server_pair

# Taken from https://github.com/fcakyon/cifar100-resnet/blob/main/cifarresnet.py

__all__ = [
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, 
        block : BasicBlock,
        num_blocks : List[int], 
        num_classes : int = 100                
    ) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class ResNetClient(nn.Module):
    def __init__(
        self, 
        block : BasicBlock, 
        in_planes : int,
        num_blocks : List[int],
    ) -> None:
        super(ResNetClient, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class ResNetServer(nn.Module):
    def __init__(
        self, 
        block : Type[BasicBlock],
        in_planes : int,
        num_blocks : List[int],
        num_classes=100,
    ) -> None:
        super(ResNetServer, self).__init__()
        self.in_planes = in_planes

        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.linear = nn.Linear(in_planes*4, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.layer2(x)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self._forward_impl(x), dim = 1)
    




def resnet20(num_classes=100):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=100):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=100):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes=100):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes=100):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(num_classes=100):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)

# ------------------------------------------------------------------------------
def resnet56_sl_client(in_planes=64, **kwargs):
    return ResNetClient(BasicBlock, num_blocks=[9, 9, 9], in_planes=in_planes)

def resnet56_sl_server(in_planes=64, num_classes=100, **kwargs):
    return ResNetServer(
        BasicBlock, num_blocks=[9, 9, 9], in_planes=in_planes,
        num_classes=num_classes
    )

register_client_server_pair('resnet56', resnet56_sl_client, resnet56_sl_server)

# ------------------------------------------------------------------------------
def resnet110_sl_client(in_planes=64, **kwargs):
    return ResNetClient(
        BasicBlock, num_blocks=[18, 18, 18], in_planes=in_planes
    )

def resnet110_sl_server(in_planes=64, num_classes=100, **kwargs):
    return ResNetServer(
        BasicBlock, num_blocks=[18, 18, 18], in_planes=in_planes,
        num_classes=num_classes
    )

register_client_server_pair(
    'resnet110', resnet110_sl_client, resnet110_sl_server
)

# ------------------------------------------------------------------------------
def resnet1202_sl_client(in_planes=16, **kwargs):
    return ResNetClient(
        BasicBlock, in_planes=in_planes, num_blocks=[200, 200, 200]
    )

def resnet1202_sl_server(in_planes=64, num_classes=100, **kwargs):
    return ResNetServer(
        BasicBlock, num_blocks=[200, 200, 200], in_planes=in_planes,
        num_classes=num_classes
    )

register_client_server_pair(
    'resnet1202', resnet1202_sl_client, resnet1202_sl_server
)

# ------------------------------------------------------------------------------