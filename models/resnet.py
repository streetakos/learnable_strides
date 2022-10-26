from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from src.strides import StrideConv2d
from src.strides import Conv2d_StridesAsInput
from src.strides import Conv2dStride2
import numpy as np
import random
import torch.nn.functional as F


__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Downsampler(nn.Module):
	
    def __init__(self, inplanes, planes):
        super().__init__()
        
        self.conv_d = Conv2d_StridesAsInput(inplanes, planes * 1, kernel_size=1 , bias=False)
        self.bn_d = nn.BatchNorm2d(planes * 1)

    def forward(self, x: Tensor, stride_h: float, stride_w: float) -> Tensor:
        out = self.conv_d(x,stride_h,stride_w)
        out = self.bn_d(out) 		
        return out

class BasicBlockStride(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes, planes,  stride = 1,   groups= 1,  base_width = 64,  dilation= 1,  norm_layer = None ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")			
			
        #dilation = 3 			
        self.conv1 = StrideConv2d(   inplanes,  planes, kernel_size=3, padding=dilation, groups=1,  bias=False, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(   planes,  planes, kernel_size=3, stride=1, padding=dilation, groups=1,  bias=False, dilation=1)		
        self.bn2 = norm_layer(planes)
        self.stride = stride
		
					
        downsample = None		
        if stride != 1 or inplanes != planes * 1:
            downsample = Downsampler(inplanes,planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
       		
        out = self.bn1(out)
        out = self.relu(out)
        '''
        w = 0 
        s = out.shape
        if self.conv2.weight.shape[3] > s[3]+1: 
            print(s,self.conv2.weight.shape)			
            w = (int(self.conv2.weight.shape[3] - s[3]),int(self.conv2.weight.shape[3] - s[3]))
            out = F.pad(out, w+(0,0), "constant", 0)			
        if self.conv2.weight.shape[2] > s[2]+1: 
            print(s,self.conv2.weight.shape)			
            w += (int(self.conv2.weight.shape[2] - s[2]),int(self.conv2.weight.shape[2] - s[2]))
            out = F.pad(out, (0,0)+w, "constant", 0)	
        ''' 	

        #kernel_size = [self.conv2.weight.shape[2],self.conv2.weight.shape[3]]
        #input_pad = (out.size(2) < kernel_size[0]) or (out.size(3) < kernel_size[1])
        
        # Add zero pad if needed		
        #if input_pad:
        #    pad_h = max(kernel_size[0] - out.size(2), 0)
        #    pad_w = max(kernel_size[1] - out.size(3), 0)
        #    out = F.pad(out, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()		
		
        #print(out.shape,self.conv2.weight.shape )
        out = self.conv2(out)
		
        #if input_pad:			
        #    out = out[:, :, :out.size(2) - pad_h, :out.size(3) - pad_w].contiguous()		
		
        out = self.bn2(out)

        if self.downsample is not None:
			
            with torch.no_grad():
			
                s = self.conv1.stride.data
                s_h = float(s[0])
                s_w = float(s[1])
            #print('s', s, s.grad) 			
            identity = self.downsample(x,s_h,s_w)

        #print('identity ', identity.shape, ' out ', out.shape)			
        out += identity
        out = self.relu(out)

        return out

class BasicBlock2Stride(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes, planes,  stride = 1,   groups= 1,  base_width = 64,  dilation= 1,  norm_layer = None ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")			
        '''			
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2dStride2(inplanes,  planes, kernel_size=3, padding=dilation, bias=False)#	conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(   planes,  planes, kernel_size=3, stride=1, padding=dilation, groups=1,  bias=False, dilation=1)
        self.bn2 = norm_layer(planes)
        self.stride = stride
		
					
        downsample = None		
        if stride != 1 or inplanes != planes * 1:
            downsample = nn.Sequential(
                Conv2dStride2(inplanes, planes, kernel_size=1, padding=dilation, bias=False)	,
                norm_layer(planes * 1),
            )
        self.downsample = downsample
        '''		
        self.conv1 = Conv2dStride2(inplanes,  planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(   planes,  planes, kernel_size=3, stride=1, padding=1, groups=1,  bias=False, dilation=1)
        self.bn2 = norm_layer(planes)
        self.stride = stride
		
       # print(stride, inplanes,planes )					
        downsample = None		
        if stride != 1:
            downsample = nn.Sequential(
                Conv2dStride2(inplanes,  planes, kernel_size=1, padding=0, bias=False),
                norm_layer(planes * 1),
            )
        #elif inplanes != planes * 1:
        #    downsample = nn.Sequential(
        #        conv1x1(inplanes, planes * 1, stride),
        #        norm_layer(planes * 1),
        #    )			
        self.downsample = downsample		
		

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)		
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)		
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
	

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes, planes,  stride = 1,   groups= 1,  base_width = 64,  dilation= 1,  norm_layer = None ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")			
			
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
		
					
        downsample = None		
        if stride != 1 or inplanes != planes * 1:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * 1, stride),
                norm_layer(planes * 1),
            )
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group         
       
        if (block == BasicBlockStride):
            self.conv1 = StrideConv2d(3, self.inplanes, kernel_size=7, padding=3, bias=False)		
        else:			
            #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1 = Conv2dStride2(3, self.inplanes, kernel_size=7, padding=3, bias=False)	
            print('OOOoook')			
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
		
        #TODO maybe change this to strided
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		
        
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
		
      
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, StrideConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, Conv2d_StridesAsInput):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")	
            elif isinstance(m, Conv2dStride2):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")					
							
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1


        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
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
        x = self.conv1(x)
        #x.register_hook(lambda grad: print('Out grad ',grad[0,0,0,0])) 		
        x = self.bn1(x)
        x = self.relu(x)		
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet( block, layers):
    model = ResNet(block, layers, num_classes=10)
    return model


def resnet18():
    return _resnet(BasicBlockStride, [2, 2, 2, 2])