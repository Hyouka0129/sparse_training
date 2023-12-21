import torch
import torch.nn as nn
import torch.nn.functional as F

sparsity_ratio=[0,0,0,0,0,0]
#sparsity ratio of CONV2D in three residual blocks, 5 levels of sparsity{0,0.25,0.5,0.75,1}

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        print("sparsity scheme:",sparsity_ratio)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out2 = self.bn3(self.conv3(out2))
        out2 += out1
        out2 = F.relu(out2)
        out3 = F.relu(self.bn4(self.conv4(out2)))
        out3 = self.bn5(self.conv5(out3))
        out3 += self.bn6(self.conv6(out2))
        out3 = F.relu(out3)
        out4 = F.relu(self.bn7(self.conv7(out3)))
        out4 = self.bn8(self.conv8(out4))
        out4 += self.bn9(self.conv9(out3))
        out4 = F.relu(out4)
        out = F.avg_pool2d(out4, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)    


# Instantiate the model
model =ResNet()

# Print the model architecture
print(model)
