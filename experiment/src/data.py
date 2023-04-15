import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from experiment.src.constants import *

def load_dataset(dataset='cifar10'):
    if(dataset=='cifar10'):
        train_dataset=datasets.CIFAR10(
            LOC+"data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ])
        )
        test_dataset=datasets.CIFAR10(
            LOC+"data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ])
        )
    elif(dataset=='gaussian'):
        x=np.random.normal(scale=GAUSSIN_SIGMA,size=(GAUSSIAN_N,GAUSSIAN_D))
        w=np.random.normal(scale=GAUSSIN_SIGMA/GAUSSIAN_D,size=GAUSSIAN_D)
        e=np.random.normal(scale=GAUSSIN_SIGMA,size=GAUSSIAN_N)
        y=np.matmul(x,w)+e
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        train_dataset=torch.utils.data.TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
        test_dataset=torch.utils.data.TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))
    return train_dataset,test_dataset

def load_data(train_dataset,test_dataset,attack=False):
    if(attack):
        train_batch=1
        test_batch=1
    else:
        train_batch=TRAIN_BATCH
        test_batch=TEST_BATCH
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, test_loader

if __name__ == "__main__":
    pass