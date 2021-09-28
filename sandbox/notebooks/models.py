import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # self.shortcut = LambdaLayer(lambda x:
                                            # F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                # self.shortcut = LambdaLayer(lambda x: torch.cat((x, x[:, :planes-in_planes, :, :]), dim=1))
                # p = int(np.ceil((planes-in_planes)/2))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, planes-in_planes), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm1d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu6(out)
        return out

class ResBlock_Q(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, BIT=16, Woption='A'):
        super(ResBlock, self).__init__()

        conv2d_q = conv2d_Q_fn(BIT)

        self.conv1 = conv2d_q(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # self.shortcut = LambdaLayer(lambda x:
                                            # F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                # self.shortcut = LambdaLayer(lambda x: torch.cat((x, x[:, :planes-in_planes, :, :]), dim=1))
                # p = int(np.ceil((planes-in_planes)/2))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, planes-in_planes), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     conv2d_q(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu6(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)

class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
    bit_width = input_bits
    min_val = -2.0
    max_val = 2.0
    scaling_impl_type = ScalingImplType.CONST # Fix the quantization range to [min_val, max_val]

class UltraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(2, 16, kernel_size=3, padding='valid')
        self.bn0 = nn.BatchNorm1d(16)
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(64, 32, kernel_size=3, padding='same')
        self.bn7 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(4064, 128)
        self.bn_dense = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 24)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        
        x = self.relu(self.bn_dense(self.fc1(x)))
        x = self.fc2(x)
        return x
