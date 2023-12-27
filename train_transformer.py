import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    #transforms.ConvertImageDtype(torch.float16),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 加载Flower102数据集
train_dataset = datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

val_dataset = datasets.Flowers102(root='./data', split='val', download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_dataset = datasets.Flowers102(root='./data', split='test', download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
num_classes = 102
# 创建ResNet模型
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
model.heads = nn.Linear(768, num_classes)
model = model.to(device)
print(model)
for name, param in model.named_parameters():
    print(name)
    print(param.shape)
    param.requires_grad = False
    if '11.self_attention' in name:
        print('reactivate')
        param.requires_grad = True
    if '10.self_attention' in name:
        print('reactivate')
        param.requires_grad = True
    if 'encoder.ln' in name:
        print('reactivate')
        param.requires_grad = True
    if 'heads' in name:
        print('reactivate')
        param.requires_grad = True
# 定义损失函数和优化器
print(count_parameters(model))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 50
#model = model.to(dtype=torch.float16)
print("training:")
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
            if '9.self_attention' in name:
                param.grad[:int(len(param.grad) // 2 )] = torch.zeros_like(param.grad[:int(len(param.grad) // 2 ) ])
            if '10.self_attention' in name:
                param.grad[:int(len(param.grad) // 2 )] = torch.zeros_like(param.grad[:int(len(param.grad) // 2 )])
            #if '11.self_attention' in name:
            #    param.grad[:len(param.grad) // 2] = torch.zeros_like(param.grad[:len(param.grad) // 2])
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
model.eval()
correct = 0
total = 0
test_loss = 0.0
with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f'test Loss: {test_loss / len(test_loader)}, Accuracy: {100 * correct / total}%')
# 保存模型
torch.save(model.state_dict(), 'resnet_flower.pth')
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
