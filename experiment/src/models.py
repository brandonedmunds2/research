import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.num_features=num_features
        self.num_classes=num_classes
        self.fc=nn.Linear(in_features=num_features,out_features=num_classes)
    def forward(self,x):
        pred=self.fc(x.view(-1,self.num_features))
        return pred
    
def simple_net(num_features,num_classes):
    return SimpleNet(num_features,num_classes)

if __name__ == "__main__":
    pass