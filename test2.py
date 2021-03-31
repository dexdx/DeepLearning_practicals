import prac2
import torch
import matplotlib.pyplot as plt
import random
from torchvision import datasets

random.seed(1)

# 1. Nearest neighbor
train1 = torch.empty((100,2)).normal_()
target1 = torch.empty(100,1).bernoulli_(1/2)
x = torch.empty(1,2).normal_()
y = prac2.nearest_classification(train1,target1,x)

"""
plt.scatter(train1[:,0],train1[:,1], c=target1, label=target1)
plt.scatter(x[0,0],x[0,1], c=y, marker='P', s=50)
plt.savefig('nearestn.png')
"""

"""
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
"""

mnist_train = datasets.MNIST('./data/', train=True, download=True)
#train_input = mnist_train.data.view(-1,1,28,28).float()
#train_input = mnist_train.data.float()
train_input = mnist_train.data.view(-1,1).float()
print(train_input.size())
train_target = mnist_train.targets
print(train_target)
print(train_input.size())
