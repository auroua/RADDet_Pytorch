from torchvision.models.resnet import Bottleneck
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, List, Optional

def build_backbone(config_model):
    radar_backbone = ResNet(block=Bottleneck, layers=config_model["blocks"], expensions=config_model["expensions"])
    return radar_backbone


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(inplanes, planes, stride, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        zero_init_residual: bool = False,
        expensions: List[bool] = None,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.layer1 = self._make_layer(block, layers[0], expension=expensions[0])
        self.layer2 = self._make_layer(block, layers[1], expension=expensions[1])
        self.layer3 = self._make_layer(block, layers[2], expension=expensions[2])
        self.layer4 = self._make_layer(block, layers[3], expension=expensions[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Bottleneck], blocks: int,  expension: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        stride = 1
        expension_flags = [1] * blocks
        if expension:
            expension_flags[-1] *= 2
        planes = self.inplanes
        layers = []

        for i in range(blocks):
            planes = planes * expension_flags[i]
            if self.inplanes != planes:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes, stride),
                    norm_layer(planes),
            )
            else:
                downsample = None
            layers.append(block(inplanes=self.inplanes, planes=planes, groups=1, stride=stride,
                                downsample=downsample, dilation=1, norm_layer=norm_layer))
            self.inplanes = planes
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == "__main__":
    import util.loader as loader
    config = loader.readConfig(config_file_name="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/config.json")
    config_model = config["MODEL"]

    radar_resnet = build_backbone(config_model)
    input = torch.randn((3, 64, 256, 256))
    out_tensor = radar_resnet(input)
