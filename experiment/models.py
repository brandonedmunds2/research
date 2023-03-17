import torch
import torchvision.models
from torchvision.models.resnet import _resnet, BasicBlock
import torch.nn as nn
import math
import torch.nn.functional as F
from constants import GN_GROUPS

class SimpleNet(nn.Module):
    def __init__(self,in_features,num_classes):
        super().__init__()
        self.fc1=nn.Linear(in_features,8)
        self.fc=nn.Linear(8,num_classes)
    def forward(self,x):
        pred=F.relu(self.fc1(x.view(-1,3*32*32)))
        return self.fc(pred)
    
def simple_net(in_features,num_classes):
    return SimpleNet(in_features,num_classes)

def batchnorm2groupnorm(model):
    for name, module in model.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(model, name)
                if isinstance(sub_layer, nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    model._modules[name] = nn.GroupNorm(GN_GROUPS, num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(model, name)
                sub_layer = batchnorm2groupnorm(sub_layer)
                model.__setattr__(name=name, value=sub_layer)
    return model

def resnet20():
    model=_resnet(BasicBlock,[3,3,3], None, True)
    batchnorm2groupnorm(model)
    return model

def resnet18():
    model=torchvision.models.resnet18()
    batchnorm2groupnorm(model)
    return model

if __name__ == "__main__":
    pass