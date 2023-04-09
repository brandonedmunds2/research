import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.num_features=num_features
        self.num_classes=num_classes
        self.fc1=nn.Linear(in_features=num_features,out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=num_classes)
    def forward(self,x):
        pred=F.relu(self.fc1(x.view(-1,self.num_features)))
        pred=self.fc2(pred)
        return pred
    
def simple_net(num_features,num_classes):
    return SimpleNet(num_features,num_classes)

if __name__ == "__main__":
    pass