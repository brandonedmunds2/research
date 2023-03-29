import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, _resnet

def wrn(num_classes=10):
    return _resnet(block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   weights=None,
                   progress=True,
                   width_per_group=64 * 2,
                   num_classes=num_classes)

class SimpleNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.fc1=nn.Linear(in_features=32*32*3,out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=num_classes)
    def forward(self,x):
        pred=F.relu(self.fc1(x.view(-1,32*32*3)))
        pred=self.fc2(pred)
        return pred
    
def simple_net(num_classes):
    return SimpleNet(num_classes)

if __name__ == "__main__":
    pass