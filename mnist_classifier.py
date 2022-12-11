import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from mnist_hebbian_weight import draw_weights


from MLP import MLP

num_workers = 0
batch_size = 20
valid_size = 0.2
lr = 0.01
n_epochs = 5
load_hebbian = 1
save_path = 'model.pt'

transform = transforms.ToTensor()

#下载数据
train_data = datasets.MNIST(root = './data',train = True,
                           download = True,transform = transform)
test_data = datasets.MNIST(root = './data',train = False,
                          download = True,transform = transform)

#创建加载器
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx,valid_idx = indices[split:],indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,
                            sampler = train_sampler,num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,
                            sampler = valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = batch_size,
                                         num_workers = num_workers)

def train(train_loader, model, n_epochs, lr, save_path='model.pt'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model.parameters(),lr = lr)

    valid_loss_min = np.Inf

    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        
        for data,target in train_loader:
            optimizer.zero_grad()
            output = model(data)#得到预测值
            
            loss = criterion(output,target)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        
        #计算检验集的损失，这里不需要反向传播
        for data,target in valid_loader:
            output = model(data)
            loss = criterion(output,target)
            valid_loss += loss.item() * data.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss))
        if valid_loss <= valid_loss_min:#保存模型
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(),save_path)
            valid_loss_min = valid_loss

def test(test_loader, model):
    criterion = nn.CrossEntropyLoss()
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

model = MLP()
print(model)

if load_hebbian:
    hebbian_weight = torch.tensor(np.loadtxt('hebbian_weights.txt')).float()
    model.fc1.weight.requires_grad = False
    model.fc1.weight.data = hebbian_weight

train(train_loader, model, n_epochs, lr, save_path)
model.load_state_dict(torch.load(save_path))
test(test_loader, model)

fc1_weight = model.fc1.weight.data.numpy()
draw_weights(fc1_weight, 10, 10, 'hebbian_MLP')