import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model_generator import ResNet

layerwise = ["conv1.weight"]#list of layerwise layer,which will be totally frozen.
channelwise = ["conv2.weight","conv3.weight"]#list of channelwise layer, selected channels will be frozen.
channel_frozen =[  [[2,2],[3,3]],    [[4,4],[5,5]]]#channel[2,2]&[3,3] of conv2 and channel[4,4]&[5,5] of conv3 will be frozen.
elementwise = ["conv4.weight"]#list of elementwise layer, selected elements will be frozen.
element_frozen = [[[2,2,2,1],[2,3,2,2]]]
#共9层卷积，即conv1-conv9
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    #transforms.ConvertImageDtype(torch.float16),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 加载Flower102数据集
train_dataset = datasets.Flowers102(root='./data', split="train", download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

val_dataset = datasets.Flowers102(root='./data', split="val", download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
num_classes = 102
# 创建ResNet模型
model = ResNet()
model = model.to(device)
model.linear = nn.Linear(64, num_classes)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#model = model.to(dtype=torch.float16)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if(name in layerwise):
                param.grad=torch.zeros_like(param.grad)
            if(name in channelwise):
                for index in channel_frozen[channelwise.index(name)]:
                    param.grad[index[0]][index[1]]=torch.zeros_like(param.grad[index[0]][index[1]])
            if(name in elementwise):
                for index in element_frozen[elementwise.index(name)]:
                    param.grad[index[0]][index[1]][index[2]][index[3]]=0
        optimizer.step()
        running_loss += loss.item()
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss / len(val_loader)}, Accuracy: {100 * correct / total}%')

print("Training finished.")

# 保存模型
torch.save(model.state_dict(), 'resnet_flower.pth')

