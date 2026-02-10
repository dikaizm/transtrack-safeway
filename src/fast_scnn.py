
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, relu_type='relu'):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu_type == 'relu' else nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class _DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(_DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class _DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(_DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return self.relu(x)

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels * t, 1, 1, 0, relu_type='relu'),
            _DWConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = _ConvBNReLU(in_channels, in_channels, 1, 1, 0)
        self.conv2 = _ConvBNReLU(in_channels, in_channels, 1, 1, 0)
        self.conv3 = _ConvBNReLU(in_channels, in_channels, 1, 1, 0)
        self.conv4 = _ConvBNReLU(in_channels, in_channels, 1, 1, 0)

        self.out = _ConvBNReLU(in_channels * 5, out_channels, 1, 1, 0)

    def forward(self, x):
        h, w = x.shape[2:]
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=True)
        return self.out(torch.cat((x, feat1, feat2, feat3, feat4), 1))

class LearningToDownsample(nn.Module):
    def __init__(self, in_channels=3, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(in_channels, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        return self.dsconv2(self.dsconv1(self.conv(x)))

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, t=6, num_blocks=(3, 3, 3)):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Conv2d(out_channels, out_channels, 1)
        self.conv_higher_res = nn.Conv2d(highter_in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(3, 32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128, 4)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.aux_classifier = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(64, num_classes, 1))

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        if self.aux and self.training:
           aux = self.aux_classifier(higher_res_features)
           aux = F.interpolate(aux, size, mode='bilinear', align_corners=True)
           return x, aux
        return x
