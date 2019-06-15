import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .sync_batchnorm import SynchronizedBatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1*atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1*atrous, dilation=atrous, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, num_classes=1000):
        super(ResNet_Atrous, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=1, atrous=atrous)
        self.layer5 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=atrous)
        self.layer6 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=atrous)
        self.layer7 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=atrous)
        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self):
        return self.layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1]*blocks
        if stride != 1 or inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=atrous[0], bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes*block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        self.layers.append(x)

        return x

def resnet50_atrous(pretrained=True, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1,2,1], **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict) 
    return model


def resnet101_atrous(pretrained=True, **kwargs):
    """Constructs a atrous ResNet-101 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[1,2,1], **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet101'])
        # print(old_dict.conv1.weight)
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        # print(model_dict)
        model.load_state_dict(model_dict) 
    return model


def resnet152_atrous(pretrained=True, **kwargs):
    """Constructs a atrous ResNet-152 model."""
    model = ResNet_Atrous(Bottleneck, [3, 8, 36, 3], atrous=[1,2,1], **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict) 
    return model

if __name__ == '__main__':
    resnet101_atrous()
