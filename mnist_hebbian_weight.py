import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def draw_weights(synapses, Kx, Ky, nep):
    fig=plt.figure(figsize=(12.9,10))
    yy=0
    HM=np.zeros((28*Ky,28*Kx))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    plt.savefig(str(nep)+'.png')
    plt.cla()

if __name__ == '__main__':
    mat = scipy.io.loadmat('mnist_all.mat')

    Nc=10
    N=784
    Ns=60000
    M=np.zeros((0,N))
    for i in range(Nc):
        M=np.concatenate((M, mat['train'+str(i)]), axis=0)
    M=M/255.0
        
    eps0=2e-2    # learning rate
    Kx=10
    Ky=10
    hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array
    mu=0.0
    sigma=1.0
    Nep=200      # number of epochs
    Num=100      # size of the minibatch
    prec=1e-30
    delta=0.4    # Strength of the anti-hebbian learning
    p=2.0        # Lebesgue norm of the weights
    k=2          # ranking parameter, must be integer that is bigger or equal than 2

    synapses = np.random.normal(mu, sigma, (hid, N))
    for nep in range(Nep):
        eps=eps0*(1-nep/Nep)
        M=M[np.random.permutation(Ns),:]
        for i in range(Ns//Num):
            inputs=np.transpose(M[i*Num:(i+1)*Num,:])
            sig=np.sign(synapses)
            tot_input=np.dot(sig*np.absolute(synapses)**(p-1),inputs)
            
            y=np.argsort(tot_input,axis=0)
            yl=np.zeros((hid,Num))
            yl[y[hid-1,:],np.arange(Num)]=1.0
            yl[y[hid-k],np.arange(Num)]=-delta
            
            xx=np.sum(np.multiply(yl,tot_input),1)
            ds=np.dot(yl,np.transpose(inputs)) - np.multiply(np.tile(xx.reshape(xx.shape[0],1),(1,N)),synapses)
            
            nc=np.amax(np.absolute(ds))
            if nc<prec:
                nc=prec
            synapses += eps*np.true_divide(ds,nc)
        if nep%60==0:
            draw_weights(synapses, Kx, Ky, nep)

    np.savetxt('hebbian_weights.txt',synapses)