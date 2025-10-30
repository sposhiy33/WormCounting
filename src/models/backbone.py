# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import src.models.vgg_ as  vgg_models
# import src.models.resnet_ as resnet_models


class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out
    
# class BackboneBase_ResNet(nn.Module):
#     def __init__(self, backbone: models_resnet.ResNet, fpn_target_channels: list):
#         super().__init__()
#         # fpn_target_channels = [c_for_features_1, c_for_features_2, c_for_features_3]
#         # e.g. [256, 512, 512], which are the channel dimensions FPN expects for its P3, P4, P5 inputs respectively.

#         # Extract ResNet layers
#         self.body0_main = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
#         self.body1_main = backbone.layer1 # Corresponds to C2 level
#         self.body2_main = backbone.layer2 # Corresponds to C3 level
#         self.body3_main = backbone.layer3 # Corresponds to C4 level
#         self.body4_main = backbone.layer4 # Corresponds to C5 level

#         # Determine input channels from ResNet layers for adapters
#         # This logic assumes Bottleneck or BasicBlock as used in standard ResNets
#         if isinstance(backbone.layer1[0], models_resnet.Bottleneck):
#             l1_channels = 256
#             l2_channels = 512
#             l3_channels = 1024
#             l4_channels = 2048
#         elif isinstance(backbone.layer1[0], models_resnet.BasicBlock):
#             l1_channels = 64
#             l2_channels = 128
#             l3_channels = 256
#             l4_channels = 512
#         else:
#             raise NotImplementedError(f"Unknown block type in ResNet: {type(backbone.layer1[0])}. Please check resnet_.py.")

#         # Adapter layers to transform ResNet layer outputs to the channel dimensions
#         # expected by the FPN (features[1], features[2], features[3]).
#         # features[1] (FPN's P3 input) is derived from ResNet's layer2 (C3) output.
#         self.adapter1 = nn.Conv2d(l2_channels, fpn_target_channels[0], kernel_size=1)
#         # features[2] (FPN's P4 input) is derived from ResNet's layer3 (C4) output.
#         self.adapter2 = nn.Conv2d(l3_channels, fpn_target_channels[1], kernel_size=1)
#         # features[3] (FPN's P5 input) is derived from ResNet's layer4 (C5) output.
#         self.adapter3 = nn.Conv2d(l4_channels, fpn_target_channels[2], kernel_size=1)

#     def forward(self, tensor_list):
#         xs = self.body0_main(tensor_list)
#         out = []

#         x1 = self.body1_main(xs) # Output of ResNet's layer1 (C2)
#         out.append(x1) # This corresponds to features[0]

#         x2 = self.body2_main(x1) # Output of ResNet's layer2 (C3)
#         out.append(self.adapter1(x2)) # Adapted, corresponds to features[1]

#         x3 = self.body3_main(x2) # Output of ResNet's layer3 (C4)
#         out.append(self.adapter2(x3)) # Adapted, corresponds to features[2]

#         x4 = self.body4_main(x3) # Output of ResNet's layer4 (C5)
#         out.append(self.adapter3(x4)) # Adapted, corresponds to features[3]
        
#         return out

# class Backbone_ResNet(BackboneBase_ResNet):
#     def __init__(self, name: str, pretrained: bool, fpn_target_channels: list = [256, 512, 512]):
#         if name == 'resnet50':
#             # Use weights from your resnet_.py, assuming it follows torchvision's API
#             weights = models_resnet.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
#             backbone_model = models_resnet.resnet50(weights=weights)
#         # Add elif for other ResNet variants like resnet18, resnet34 if needed
#         # elif name == 'resnet18':
#         #     weights = models_resnet.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#         #     backbone_model = models_resnet.resnet18(weights=weights)
#         else:
#             raise ValueError(f"Unsupported ResNet backbone: {name}")
        
#         super().__init__(backbone_model, fpn_target_channels)
#         # num_channels is the channel dimension of the FPN output features (P3, P4, P5)
#         # that are fed to the detection/segmentation heads.
#         # Your FPN's `feature_size` is 256.
#         self.num_channels = 256

# class BackboneBase_ResNet(nn.Module):
#     """ResNet backbone with frozen BatchNorm."""
#     pass

class Backbone_VGG(BackboneBase_VGG):
    """VGG backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = vgg_models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = vgg_models.vgg16(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def build_backbone(args):
    backbone = Backbone_VGG(args.backbone, True)
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg16', True)
