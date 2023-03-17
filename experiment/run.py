import numpy as np
import torch
import random
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models import resnet20, simple_net, resnet18
from sklearn.model_selection import train_test_split
from constants import *
from cust_opacus import CustomPrivacyEngine
from prune import prune_grads, prune_mask

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, model, optimizer, criterion,masks=None):
    model=model.train()
    losses = []
    correct=0
    incorrect=0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        prune_grads(optimizer,masks)
        optimizer.step()
        losses.append(loss.item())
        correct += torch.sum(output.argmax(axis=1) == target)
        incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def test(test_loader, model, criterion):
    model=model.eval()
    losses = []
    correct = 0
    incorrect=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            correct += torch.sum(output.argmax(axis=1) == target)
            incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def load_data():
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
    private_dataset, public_dataset=train_test_split(train_dataset,test_size=0.05)
    private_loader = torch.utils.data.DataLoader(
        dataset=private_dataset,
        batch_size=PRIVATE_HYPERPARAMS["train_batch"],
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    public_loader = torch.utils.data.DataLoader(
        dataset=public_dataset,
        batch_size=PUBLIC_HYPERPARAMS["train_batch"],
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=TEST_BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return private_loader, public_loader, test_loader

def plt_losses(train_losses,test_losses,epochs):
    plt.figure()
    plt.plot(range(epochs),train_losses, label="Train Loss")
    plt.plot(range(epochs),test_losses, label="Test Loss")
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def train_test(model,train_loader,test_loader,optimizer,criterion, hyperparams,masks=None,privacy_engine=None):
    train_losses=[]
    test_losses=[]
    for epoch in range(hyperparams["epochs"]):
        train_loss,train_acc=train(train_loader,model,optimizer,criterion,masks)
        train_losses.append(train_loss)
        test_loss,test_acc=test(test_loader,model,criterion)
        test_losses.append(test_loss)
        if(privacy_engine!=None):
            epsilon=privacy_engine.accountant.get_epsilon(delta=DELTA)
        else:
            epsilon=-1
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}, Epsilon: {epsilon}')
    # plt_losses(train_losses,test_losses,hyperparams["epochs"])

def main():
    private_loader,public_loader,test_loader=load_data()
    model=resnet18()
    # model=resnet20()
    # model=simple_net(3*32*32,NUM_CLASSES)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    print("Public")
    optimizer=optim.Adam(model.parameters(),lr=PUBLIC_HYPERPARAMS["lr"])
    train_test(model,public_loader,test_loader,optimizer,criterion,PUBLIC_HYPERPARAMS)
    print("Prune")
    masks=prune_mask(model,PRUNE_TYPE,PRUNE_AMOUNT,PRUNE_LARGEST)
    print("Private")
    optimizer=optim.Adam(model.parameters(),lr=PRIVATE_HYPERPARAMS["lr"])
    privacy_engine = CustomPrivacyEngine(masks=masks,secure_mode=False)
    model=model.train()
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=private_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
    )
    model=model.to(device)
    train_test(model,private_loader,test_loader,optimizer,criterion,PRIVATE_HYPERPARAMS,masks,privacy_engine)

if __name__ == "__main__":
    main()