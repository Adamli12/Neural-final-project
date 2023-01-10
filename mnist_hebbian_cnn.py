import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class RePU(nn.ReLU):
    def __init__(self, n):
        super(RePU, self).__init__()
        self.n = n

    def forward(self, x: torch.Tensor):
        return torch.relu(x) ** self.n
# def hebbian_rule(inputs,w,y,c=0.1):
#     d_ws = torch.zeros(inputs.size(0))
#     for idx, x in enumerate(inputs):
#         y = torch.dot(w, x)

#         d_w = torch.zeros(w.shape)
#         for i in range(y.shape[0]):
#             for j in range(x.shape[0]):
#                 d_w[i, j] = c * x[j] * y[i]

#         d_ws[idx] = d_w
#     return d_w

def hebbian_rule(inputs: torch.Tensor, weights: torch.Tensor):
    precision=1e-30
    delta=0.4
    norm=2
    k=2
    batch_size = inputs.shape[0]
    num_hidden_units = weights.shape[0]
    input_size = inputs[0].shape[0]
    inputs = torch.t(inputs)
    tot_input = torch.matmul(torch.sign(weights) * torch.abs(weights) ** (1), inputs)
    _, indices = torch.topk(tot_input, k, dim=0)

    activations = torch.zeros((num_hidden_units, batch_size))
    activations[indices[0], torch.arange(batch_size)] = 1.0
    activations[indices[k - 1], torch.arange(batch_size)] = delta
    xx = torch.sum(torch.mul(activations, tot_input), 1)
    norm_factor = torch.mul(xx.view(xx.shape[0], 1).repeat((1, input_size)), weights)
    ds = torch.matmul(activations, torch.t(inputs)) - norm_factor
    nc = torch.max(torch.abs(ds))
    if nc < precision:
        nc = precision
    d_w = torch.true_divide(ds, nc)

    return d_w


def create_conv1_model(input_dim, input_channels=1, num_kernels=8, kernel_size=5, pool_size=2, n=1, batch_norm=False,
                       dropout=None):
    modules = [
        ('conv1', nn.Conv2d(input_channels, num_kernels, kernel_size, bias=False))
    ]

    if batch_norm:
        modules.append(('batch_norm', nn.BatchNorm2d(num_features=num_kernels)))

    modules.extend([
        ('repu', RePU(n)),
        ('pool1', nn.MaxPool2d(pool_size)),
    ])

    if dropout is not None:
        modules.append(('dropout1', nn.Dropout2d(dropout)))

    modules.extend([
        ('flatten', Flatten()),
        ('linear1', nn.Linear(num_kernels * int(((input_dim - (kernel_size - 1)) / 2)) ** 2, 10))
    ])

    return nn.Sequential(OrderedDict(modules))


# def draw_weights(synapses, Kx, Ky, nep):
#     fig=plt.figure(figsize=(12.9,10))
#     yy=0
#     HM=np.zeros((28*Ky,28*Kx))
#     for y in range(Ky):
#         for x in range(Kx):
#             HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
#             yy += 1
#     plt.clf()
#     nc=np.amax(np.absolute(HM))
#     im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
#     fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
#     plt.axis('off')
#     plt.savefig(str(nep)+'.png')
#     plt.cla()
    
def train(train_loader, model, n_epochs, lr, save_path='model.pt'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model.parameters(),lr = lr)
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
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss))
    

def test(test_loader, model):
    criterion = nn.CrossEntropyLoss()
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(20):
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


    
    

def hebbian_weight(shape):
    mat = scipy.io.loadmat('mnist_all.mat')
    Nc=10
    Ns=60000
    M=np.zeros((0,784))
    for i in range(Nc):
        M=np.concatenate((M, mat['train'+str(i)]), axis=0)
    M=M/255.0
        
    eps0=2e-2    # learning rate
    mu=0.0
    sigma=1.0
    Nep=10      # number of epochs
    Num=100      # size of the minibatch

    synapses = np.random.normal(mu, sigma,shape)
    for nep in range(Nep):
        eps=eps0*(1-nep/Nep)
        M=M[np.random.permutation(Ns),:]
        for i in range(Ns//Num):
            inputs=np.transpose(M[i*Num:(i+1)*Num,:])
            weights = np.reshape(synapses,(-1,1))
            synapses += eps*(hebbian_rule(torch.Tensor(inputs),torch.Tensor(weights)).view(shape))

    # np.savetxt('hebbian_weights.txt',synapses)
    print(synapses.shape)
    return synapses
    
    # begin cnn
    
model = create_conv1_model(28, 1, num_kernels=400, n=1, batch_norm=True)
#下载数据
transform = transforms.ToTensor()
train_data = datasets.MNIST(root = './data',train = True,
                        download = True,transform = transform)
test_data = datasets.MNIST(root = './data',train = False,
                        download = True,transform = transform)
train_loader = torch.utils.data.DataLoader(train_data,batch_size = 20)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = 20)

conv1_shape = model.conv1.weight.shape
hebbian_trained_weight = hebbian_weight(conv1_shape)
model.conv1.weight.data = hebbian_trained_weight
model.conv1.weight.requires_grad = False

train(train_loader, model, 10, 1e-3)
test(test_loader, model)




