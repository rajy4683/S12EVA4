"""
     All model class definitions for MNIST
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class MNISTDigitBuilder(nn.Module):
    def __init__(self, dropout=0.1):
        super(MNISTDigitBuilder, self).__init__()
        self.dropout_val = dropout
        self.bias = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Conv2d(8, 8, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(8),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1,stride=1, bias=self.bias),            
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 16, 3, padding=1, bias=self.bias),            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val),
            # nn.Conv2d(16, 16, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            #nn.Dropout(self.dropout_val)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3,bias=self.bias),            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(16, 16, 3,bias=self.bias),            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val)
        )
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(16, 10, 1, bias=self.bias)
        )
                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BareCIFAR(nn.Module):
    def __init__(self):
        super(BareCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFARModelDepthDilate(nn.Module):
    def __init__(self,dropout=0.1):
        super(CIFARModelDepthDilate, self).__init__()
        self.layer1_channels = 32
        self.dropout_val = dropout
        self.bias = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias),
            #nn.Conv2d(3,self.layer1_channels,1,1,0,1,1,bias=bias),
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias,groups=self.layer1_channels),
            nn.Conv2d(self.layer1_channels,self.layer1_channels,1,1,0,1,1,bias=self.bias),      
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layer1_channels, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias), #groups=self.layer1_channels),
            #nn.Conv2d(self.layer1_channels,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels*2),
            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),      
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias,dilation=2), #groups=self.layer1_channels*2),
            #nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),       
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias, groups=self.layer1_channels*4, dilation=2),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),#, groups=self.layer1_channels),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(self.layer1_channels*8, 10, 1, bias=self.bias)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)

        return x

class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.layer1_channels = 32
        self.dropout_val = 0.1
        self.bias = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layer1_channels, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*4, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(self.layer1_channels*8, 10, 1, bias=self.bias)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F






class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,dropout=0.0 ):
        super(BasicBlock, self).__init__()
        self.dropout_val=dropout
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                #nn.Dropout(self.dropout_val)
            )
        self.dropout_layer = nn.Dropout(self.dropout_val)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))#self.dropout_layer(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))#self.dropout_layer(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,dropout=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,dropout=self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(dropout=0.0, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], dropout=dropout, num_classes=num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

"""
    ModifiedResBlock: Class for creating Modified ResNet block. Based on S11:
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU
        R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) 
        Add(X, R1)

"""

class ModifiedResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ModifiedResBlock, self).__init__()
        self.layerconv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )
        ### This layer applies after the first conv and we intend to keep the channel size same
        self.resconv = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        #self.shortcut = nn.Sequential() 

    def forward(self, x):
        out = self.layerconv(x)
        res = self.resconv(out)
        #out = res
        #out = F.relu(out)
        return out+res

"""
    S11: Custom resnet block based model
    It used the ModifiedResBlock which doesnt have multiple layers.
    PrepLayer:
        Conv 3x3 s1, p1) >> BN >> RELU [64]
    Layer1:
        ModifiedResBlock(128)
    Layer 2:
        Conv 3x3 [256]
        MaxPooling2D
        BN
        ReLU
    Layer 3:
        ModifiedResBlock(512)
    MaxPooling:(with Kernel Size 4) 
    FC Layer 
    SoftMax
"""

class S11ResNet(nn.Module):
    def __init__(self, num_classes=10,dropout=0.0):
        super(S11ResNet, self).__init__()
        self.in_planes = 64
        self.resize_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.prep_layer = nn.Sequential(
            nn.Conv2d(64, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU()
        )
        self.layer1 = ModifiedResBlock(self.in_planes, self.in_planes*2, 1)
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.in_planes*2, self.in_planes*4, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.in_planes*4),
            nn.ReLU()
        )
        self.layer3 = ModifiedResBlock(self.in_planes*4, self.in_planes*8, 1)
        self.layer4_bigmax = nn.MaxPool2d(4,4)
        #self.fc_layer = nn.Linear(512, 10)
        self.fc_layer = nn.Linear(512, 200)

    def forward(self, x):
        out = self.resize_layer(x)
        out = self.prep_layer(out)
        #out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4_bigmax(out)
        out = out.view(-1, 512)
        #out = out.view(-1, 10)
        out = self.fc_layer(out)
        #
        out = F.log_softmax(out, dim=1)
        return out




# test()
