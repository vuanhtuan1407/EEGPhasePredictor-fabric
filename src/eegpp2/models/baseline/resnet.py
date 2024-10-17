from torch import nn


def conv3x(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


# use for resnet18, resnet34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x(out_channels, out_channels * self.expansion)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        # outx = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            out = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


# use for resnet50, resnet101, resnet152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # base_channels = int(out_channels * base_width / 64.0) * group
        base_channels = int(out_channels * 64 / 64) * 1
        self.conv1 = conv1x(in_channels, base_channels)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x(base_channels, base_channels, stride)
        self.bn2 = nn.BatchNorm1d(base_channels)
        self.conv3 = conv1x(base_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        # outx = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, in_channels, num_block_list, num_classes=1000):
        super(ResNet, self).__init__()
        self.init_in_channels = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, num_block_list[0], out_channels=64)
        self.layer2 = self._make_layer(ResBlock, num_block_list[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, num_block_list[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, num_block_list[3], out_channels=512, stride=2)

        self.adap_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adap_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _make_layer(self, ResBlock, num_block, out_channels, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.init_in_channels != out_channels * ResBlock.expansion:
            downsample = nn.Sequential(
                conv1x(self.init_in_channels, out_channels * ResBlock.expansion, stride),
                nn.BatchNorm1d(out_channels * ResBlock.expansion),
            )
        layers.append(ResBlock(self.init_in_channels, out_channels, stride, downsample))
        self.init_in_channels = out_channels * ResBlock.expansion

        for _ in range(num_block - 1):
            layers.append(ResBlock(self.init_in_channels, out_channels))

        return nn.Sequential(*layers)


# use for embedding: set num_classes = 128 or 512
class ResNet18(ResNet):
    def __init__(self, in_channels, num_classes=128, pretrained=False):
        super(ResNet18, self).__init__(BasicBlock, in_channels, [2, 2, 2, 2], num_classes)


class ResNet34(ResNet):
    def __init__(self, in_channels, num_classes=128, pretrained=False):
        super(ResNet34, self).__init__(BasicBlock, in_channels, [3, 4, 6, 3], num_classes)


class ResNet50(ResNet):
    def __init__(self, in_channels, num_classes=128, pretrained=False):
        super(ResNet50, self).__init__(Bottleneck, in_channels, [3, 4, 6, 3], num_classes)


class ResNet101(ResNet):
    def __init__(self, in_channels, num_classes=128, pretrained=False):
        super(ResNet101, self).__init__(Bottleneck, in_channels, [3, 4, 23, 3], num_classes)


class ResNet152(ResNet):
    def __init__(self, in_channels, num_classes=128, pretrained=False):
        super(ResNet152, self).__init__(Bottleneck, in_channels, [3, 8, 36, 3], num_classes)
