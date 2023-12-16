import torch
import torch.nn as nn
import torch.nn.functional as F

sparsity_ratio=[0.5,0.5,0.5,0.5,0.5,0.5]
#sparsity ratio of CONV2D in three residual blocks, 5 levels of sparsity{0,0.25,0.5,0.75,1}

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, sparsity1,sparsity2,in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv11= nn.Conv2d(in_channels, int(out_channels/4), kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv12= nn.Conv2d(in_channels, int(out_channels/4), kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv13= nn.Conv2d(in_channels, int(out_channels/4), kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv14= nn.Conv2d(in_channels, int(out_channels/4), kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv21= nn.Conv2d(out_channels, int(out_channels/4), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv22= nn.Conv2d(out_channels, int(out_channels/4), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv23= nn.Conv2d(out_channels, int(out_channels/4), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv24= nn.Conv2d(out_channels, int(out_channels/4), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv11.requires_grad_(False)
        if sparsity1>=0.25:
            self.conv11.requires_grad_(False)
            if sparsity1>=0.5:
                self.conv12.requires_grad_(False)
                if sparsity1>=0.75:
                    self.conv13.requires_grad_(False)
                    if sparsity1>=1:
                        self.conv14.requires_grad_(False)
        if sparsity2>=0.25:
            self.conv21.requires_grad_(False)
            if sparsity2>=0.5:
                self.conv22.requires_grad_(False)
                if sparsity2>=0.75:
                    self.conv23.requires_grad_(False)
                    if sparsity2>=1:
                        self.conv24.requires_grad_(False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = torch.cat([self.conv11(x),self.conv12(x),self.conv13(x),self.conv14(x)],dim=1)
        out=F.relu(self.bn1(out))
        out = self.bn2(torch.cat([self.conv21(out),self.conv22(out),self.conv23(out),self.conv24(out)],dim=1))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        print("sparsity scheme:",sparsity_ratio)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.requires_grad_(False)#in mcunet starting layers are fixed
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,sparsity1=sparsity_ratio[0],sparsity2=sparsity_ratio[1])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,sparsity1=sparsity_ratio[2],sparsity2=sparsity_ratio[3])
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,sparsity1=sparsity_ratio[4],sparsity2=sparsity_ratio[5])
        self.linear = nn.Linear(64, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride,sparsity1,sparsity2):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(sparsity1=sparsity1,sparsity2=sparsity2,in_channels=self.in_channels,out_channels=out_channels,stride=stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)    


# Instantiate the model
model =ResNet(ResidualBlock, [2,2,2])

# Print the model architecture
print(model)
