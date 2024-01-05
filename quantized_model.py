import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
class MyViTModel(nn.Module):
    def __init__(self, vit_model):
        super(MyViTModel, self).__init__()
        self.vit_model = vit_model
    def forward(self, x):
        x = self.vit_model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit_model.encoder.dropout(x)
        x += self.vit_model.encoder.pos_embedding
        for i in range(12):
            layer_name = f"self.vit_model.encoder.layers.encoder_layer_{i}"
            y = eval(f"{layer_name}.ln_1")(x)
            y =torch.round(y*16)
            #print(torch.max(y))
            #print(torch.max(eval(f"{layer_name}.self_attention.in_proj_weight")))
            qkv = torch.matmul(y, torch.transpose(eval(f"{layer_name}.self_attention.in_proj_weight"), 0, 1)) + eval(f"{layer_name}.self_attention.in_proj_bias")
            qkv = qkv/4096
            #print(torch.max(qkv))
            q = qkv[:, :, :768]
            k = qkv[:, :, 768:1536]
            v = qkv[:, :, 1536:2304]
            q = torch.transpose(q.view(q.size(0), q.size(1), 12, 64), 1, 2)
            k = torch.transpose(k.view(k.size(0), k.size(1), 12, 64), 1, 2)
            v = torch.transpose(v.view(v.size(0), v.size(1), 12, 64), 1, 2)
            score = torch.matmul(q, torch.transpose(k, 2, 3)) / 8.0
            score = F.softmax(score, dim=-1)
            out = torch.transpose(torch.matmul(score, v), 1, 2)
            out = out.reshape(out.size(0), out.size(1), 768)
            out = torch.round(out*64)
            #print(torch.max(out))
            out = eval(f"{layer_name}.self_attention.out_proj")(out)
            out = out/16384
            y = eval(f"{layer_name}.dropout")(out)
            y = y + x
            x = eval(f"{layer_name}.ln_2")(y)
            x = eval(f"{layer_name}.mlp")(x)
            x = x + y
        x = x[:, 0]
        x = self.vit_model.heads(x)
        return x
model = MyViTModel(models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1))


