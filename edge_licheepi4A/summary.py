import torch
from models import *
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = VGG16().to(device)
#需要重写summary
summary(vgg, (3, 224, 224))
