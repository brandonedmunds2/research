import numpy as np
import torch
import random
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from opacus.utils.batch_memory_manager import BatchMemoryManager
try:
    from experiment.src.constants import *
except:
    import sys
    sys.path.insert(1, "C:\\Users\\brand\\research")
    from experiment.src.constants import *
from experiment.src.models import simple_net
from experiment.src.cust_opacus import MaskedPrivacyEngine
from experiment.src.mia import attack
from experiment.src.prune import prune_mask

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

def load_data(attack=False):
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

def plt_losses(train_losses,test_losses,epochs):
    plt.figure()
    plt.plot(range(epochs),train_losses, label="Train Loss")
    plt.plot(range(epochs),test_losses, label="Test Loss")
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def train_test(model,train_loader,test_loader,optimizer,criterion,epochs,private):
    train_losses=[]
    test_losses=[]
    for epoch in range(epochs):
        if(private):
            with BatchMemoryManager(
                data_loader=train_loader, max_physical_batch_size=PHYSICAL_BATCH, optimizer=optimizer
            ) as train_loader:
                train_loss,train_acc=train(train_loader,model,optimizer,criterion)
        else:
            train_loss,train_acc=train(train_loader,model,optimizer,criterion)
        train_losses.append(np.mean(train_loss))
        test_loss,test_acc=test(test_loader,model,criterion)
        test_losses.append(np.mean(test_loss))
        print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss)}, Test Loss: {np.mean(test_loss)}, Train Acc: {train_acc}, Test Acc: {test_acc}')
    plt_losses(train_losses,test_losses,epochs)
    return np.array(train_loss), np.array(test_loss)

def main(prune_layers=('fc1','fc2'),prune_amount=0.5):
    print_constants()
    train_loader,test_loader=load_data()
    model=simple_net(32*32*3,NUM_CLASSES)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=LR)
    model=model.to(device)
    # train_test(model,train_loader,test_loader,optimizer,criterion,PRE_EPOCHS,False)
    masks=prune_mask(model,PRUNE_TYPE,prune_layers,prune_amount,PRUNE_LARGEST)
    model=model.train()
    privacy_engine = MaskedPrivacyEngine(masks=masks,secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM
    )
    model=model.to(device)
    train_test(model,train_loader,test_loader,optimizer,criterion,EPOCHS,True)
    train_loader,test_loader=load_data(attack=True)
    test_loss,test_acc=test(test_loader,model,criterion)
    train_loss,train_acc=test(train_loader,model,criterion)
    print(f'Test Acc: {test_acc}')
    attack(np.array(train_loss),np.array(test_loss))

if __name__ == "__main__":
    main()