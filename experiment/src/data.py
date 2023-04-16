import torch
import numpy as np
from torchvision import datasets, transforms
from experiment.src.constants import *

def load_dataset(dataset='cifar10'):
    if(dataset=='cifar10'):
        train_dataset=datasets.CIFAR10(
            LOC+"data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                transforms.Resize(CIFAR10_SHAPE),
                transforms.Grayscale()
            ])
        )
        test_dataset=datasets.CIFAR10(
            LOC+"data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                transforms.Resize(CIFAR10_SHAPE),
                transforms.Grayscale()
            ])
        )
        ii = list(np.where(np.array(train_dataset.targets) == train_dataset.class_to_idx['dog'])[0])
        ii.extend(list(np.where(np.array(train_dataset.targets) == train_dataset.class_to_idx['cat'])[0]))
        train_dataset=torch.utils.data.Subset(train_dataset,np.random.choice(ii,CIFAR10_SAMPLES,replace=False))
        ii = list(np.where(np.array(test_dataset.targets) == test_dataset.class_to_idx['dog'])[0])
        ii.extend(list(np.where(np.array(test_dataset.targets) == test_dataset.class_to_idx['cat'])[0]))
        test_dataset=torch.utils.data.Subset(test_dataset,ii)
    elif(dataset=='mnist'):
        train_dataset=datasets.MNIST(
            LOC+"data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD)
            ])
        )
        test_dataset=datasets.MNIST(
            LOC+"data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD)
            ])
        )
        train_dataset=torch.utils.data.Subset(train_dataset,np.random.choice(len(train_dataset),MNIST_SAMPLES,replace=False))
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