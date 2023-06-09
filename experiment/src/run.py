import numpy as np
import torch
import pandas as pd
import random
from torch import nn, optim
import matplotlib.pyplot as plt
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
from experiment.src.data import load_data, load_dataset

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
        with BatchMemoryManager(
            data_loader=train_loader, max_physical_batch_size=PHYSICAL_BATCH, optimizer=optimizer
        ) as train_loader:
            train_loss,train_acc=train(train_loader,model,optimizer,criterion)
        train_losses.append(np.mean(train_loss))
        test_loss,test_acc=test(test_loader,model,criterion)
        test_losses.append(np.mean(test_loss))
        if(VERBOSE>0):
            print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss)}, Test Loss: {np.mean(test_loss)}, Train Acc: {train_acc}, Test Acc: {test_acc}')
    if(VERBOSE>0):
        plt_losses(train_losses,test_losses,EPOCHS)
    return np.array(train_loss), np.array(test_loss)

def run_instance(amount=0.0,strategy='magnitude'):
    train_dataset,test_dataset=load_dataset(dataset='cifar10')
    train_loader,test_loader=load_data(train_dataset,test_dataset)
    model=simple_net(8*8,HIDDEN_SIZE,NUM_CLASSES)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=LR)
    privacy_engine = MaskedPrivacyEngine(amount=amount,strategy=strategy,secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM
    )
    model=model.to(device)
    train_test(model,train_loader,test_loader,optimizer,criterion)
    train_loader,test_loader=load_data(train_dataset,test_dataset,attack=True)
    test_loss,test_acc=test(test_loader,model,criterion)
    train_loss,train_acc=test(train_loader,model,criterion)
    attack_auc=attack(np.array(train_loss),np.array(test_loss))
    return test_acc, attack_auc

def main(params=[(0.0,'magnitude')]):
    columns=['Description','Param','Test Acc (%)','Attack AUC']
    df=pd.DataFrame(columns=columns)
    for p in params:
        print(p)
        test_acc,attack_auc=run_instance(*p)
        new={'Description':p[1],'Param':p[0],'Test Acc (%)':test_acc.item(),'Attack AUC':attack_auc}
        df=df.append(new,ignore_index=True)
    print(df)
    df.to_csv(LOC+'result.csv',index=False)

if __name__ == "__main__":
    main()