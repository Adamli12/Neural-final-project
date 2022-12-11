import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self):
    super(MLP,self).__init__()
    
	#两个全连接的隐藏层，一个输出层
 	#因为图片是28*28的，需要全部展开，最终我们要输出数字，一共10个数字。
 	#10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
    hidden_1 = 100
    hidden_2 = 100
    self.fc1 = nn.Linear(28 * 28,hidden_1)
    self.fc2 = nn.Linear(hidden_1,hidden_2)
    self.fc3 = nn.Linear(hidden_2,10)
    #使用dropout防止过拟合
    self.dropout = nn.Dropout(0.2)
  def forward(self,x):
    x = x.view(-1,28 * 28)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    
    x = F.relu(self.fc2(x))
    
    x = self.dropout(x)
    x = self.fc3(x)
#     x = F.log_softmax(x,dim = 1)
    
    return x