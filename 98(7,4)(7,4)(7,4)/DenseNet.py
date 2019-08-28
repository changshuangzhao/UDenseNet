import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from torchvision.models import squeezenet

class BasicBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, drop_rate):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_planes),
                                  nn.ReLU(True),
                                  nn.Conv2d(in_planes, growth_rate, 3, 1, 1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, block, in_planes, growth_rate, nb_block, drop_rate):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_block, drop_rate)
        self.in_planes = in_planes + nb_block * growth_rate

        self.conv = nn.Sequential(nn.BatchNorm2d(self.in_planes),
                                  nn.ReLU(True),
                                  nn.Conv2d(self.in_planes, self.in_planes, 1, 1, 0, bias=False))

    def _make_layer(self, block, in_planes, growth_rate, nb_block, drop_rate):
        layers = []
        for i in range(nb_block):
            layers.append(block(in_planes + i * growth_rate, growth_rate, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv(out)

        return out


class DenseNet(nn.Module):
    def __init__(self, num_classes=10, growth_rate=12, nb_layer1=4, nb_layer2=4, nb_layer3=4, nb_block1=7, nb_block2=7, nb_block3=7, drop_rate=0.0):
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        block_growth_rate1 = growth_rate * nb_block1
        block_growth_rate2 = growth_rate * nb_block2
        block_growth_rate3 = growth_rate * nb_block3

        self.conv1 = nn.Conv2d(3, in_planes, 3, 1, 1, bias=False)

        self.block1 = self._make_layer(DenseBlock, in_planes, growth_rate, block_growth_rate1, nb_layer1, nb_block1, drop_rate)
        in_planes = int(in_planes + nb_layer1 * block_growth_rate1)
        
        self.block2 = self._make_layer(DenseBlock, in_planes, growth_rate, block_growth_rate2, nb_layer2, nb_block2, drop_rate)
        in_planes = int(in_planes + nb_layer2 * block_growth_rate2)
        
        self.block3 = self._make_layer(DenseBlock, in_planes, growth_rate, block_growth_rate3, nb_layer3, nb_block3, drop_rate)
        in_planes = int(in_planes + nb_layer3 * block_growth_rate3)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = F.avg_pool2d(self.block1(out), 2)
        out = F.avg_pool2d(self.block2(out), 2)
        out = self.block3(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

    def _make_layer(self, block, in_planes, growth_rate, block_growth_rate, nb_denseblock, nb_block, drop_rate):
        layers = []
        for i in range(nb_denseblock):
            layers.append(
                block(BasicBlock, in_planes + i * block_growth_rate, growth_rate, nb_block, drop_rate))
        return nn.Sequential(*layers)

