# ------------------------------------------------------------------------------
from models import aux_models
from typing import Type, Union, List, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3
from models.aux_models import register_auxiliary_model

# ------------------------------------------------------------------------------
class ResNetAuxiliary(aux_models.GradScalarAuxiliaryModel):
    def __init__(
        self,
        server,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_planes: int = 128,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device='cpu', align_epochs=5, align_step=1e-3,
        align_batch_size=100, max_dataset_size=1000
    ) -> None:
        super(ResNetAuxiliary, self).__init__(
            server, device, align_epochs, align_batch_size, max_dataset_size
        )

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = in_planes   # TODO: changed
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        #self.layer2 = self._make_layer(
        #    block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        #)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block
        # behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        self.align_epochs = align_epochs

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        #x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_inner(self, x: Tensor) -> Tensor:
        return F.log_softmax(self._forward_impl(x), dim=1)

# ------------------------------------------------------------------------------
def _resnet_sl_auxiliary(
    server,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    in_planes: int,
    weights: Optional[Any],
    progress: bool,
    device: str = 'cpu',
    align_epochs=5, align_step=1e-3,
    align_batch_size=100, max_dataset_size=1000,
    **kwargs: Any,
) -> List[ResNetAuxiliary]:

    aux_model = ResNetAuxiliary(
        server, block, layers, in_planes, device=device,
        align_epochs=align_epochs, align_step=align_step,
        align_batch_size=align_batch_size, max_dataset_size=max_dataset_size,
        **kwargs
    )
    if weights is not None:
        aux_model.load_state_dict(
            weights[2].get_state_dict(progress=progress, check_hash=True)
        )

    return aux_model

# ------------------------------------------------------------------------------
@register_auxiliary_model("resnet18", disable_check=True)
def resnet18_sl_aux(server, layers=None, in_planes: int = 128,
    weights: Optional[Any] = None, progress: bool = True, num_classes: int = 10,
    device='cpu', **kwargs: Any
):
    """ResNet-18 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    if layers is None: layers = [2, 2, 2, 2]
    return _resnet_sl_auxiliary(
        server, BasicBlock, layers, in_planes, weights, progress,
        num_classes=num_classes, device=device, **kwargs
    )

# ------------------------------------------------------------------------------
@register_auxiliary_model("resnet50", disable_check=True)
def resnet50_sl_aux(server, layers=None, in_planes: int=512,
    weights: Optional[Any] = None, progress: bool = True, num_classes: int = 10,
    device='cpu', **kwargs: Any
):
    """ResNet-18 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    if layers is None: layers = [3, 4, 6, 3]
    return _resnet_sl_auxiliary(
        server, BasicBlock, layers, in_planes, weights, progress,
        num_classes=num_classes, device=device, **kwargs
    )

# ------------------------------------------------------------------------------
@register_auxiliary_model("resnet152", disable_check=True)
def resnet152_sl_aux(server, layers=None, in_planes: int=512,
    weights: Optional[Any] = None, progress: bool = True, num_classes: int = 10,
    device='cpu', **kwargs: Any
):
    """ResNet-18 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    if layers is None: layers = [3, 8, 36, 3]
    return _resnet_sl_auxiliary(
        server, BasicBlock, layers, in_planes, weights, progress,
        num_classes=num_classes, device=device, **kwargs
    )
