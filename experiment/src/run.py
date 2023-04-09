import numpy as np
import torch
import random
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models import simple_net
from constants import *
from opacus.utils.batch_memory_manager import BatchMemoryManager
from cust_opacus import CustomPrivacyEngine
from mia import attack
from prune import prune_mask

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(train_loader, model, optimizer, criterion):
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
        optimizer.step()
        losses.append(loss.item())
        correct += torch.sum(output.argmax(axis=1) == target)
        incorrect += torch.sum(output.argmax(axis=1) != target)
    return losses, (100.0 * correct / (correct+incorrect))

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
    return losses, (100.0 * correct / (correct+incorrect))

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
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH,
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
    return train_loader, test_loader

def plt_losses(train_losses,test_losses,epochs):
    plt.figure()
    plt.plot(range(epochs),train_losses, label="Train Loss")
    plt.plot(range(epochs),test_losses, label="Test Loss")
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def train_test(model,train_loader,test_loader,optimizer,criterion):
    train_losses=[]
    test_losses=[]
    for epoch in range(EPOCHS):
        # with BatchMemoryManager(
        #     data_loader=train_loader, max_physical_batch_size=PHYSICAL_BATCH, optimizer=optimizer
        # ) as train_loader:
        train_loss,train_acc=train(train_loader,model,optimizer,criterion)
        train_losses.append(np.mean(train_loss))
        test_loss,test_acc=test(test_loader,model,criterion)
        test_losses.append(np.mean(test_loss))
        print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss)}, Test Loss: {np.mean(test_loss)}, Train Acc: {train_acc}, Test Acc: {test_acc}')
    plt_losses(train_losses,test_losses,EPOCHS)
    return np.array(train_loss), np.array(test_loss)

def main():
    train_loader,test_loader=load_data()
    model=simple_net(32*32*3,NUM_CLASSES)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=LR)
    # privacy_engine = CustomPrivacyEngine()
    # model, optimizer, train_loader = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     noise_multiplier=NOISE_MULTIPLIER,
    #     max_grad_norm=MAX_GRAD_NORM,
    # )
    model=model.to(device)
    train_test(model,train_loader,test_loader,optimizer,criterion)
    masks=prune_mask(model,PRUNE_TYPE,PRUNE_LAYERS,PRUNE_AMOUNT,PRUNE_LARGEST)
    test_loss,test_acc=test(test_loader,model,criterion)
    train_loss,test_acc=test(train_loader,model,criterion)
    attack(train_loss,test_loss)

if __name__ == "__main__":
    main()