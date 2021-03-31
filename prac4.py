import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

############################################################################

class Net(nn.Module):
    def __init__(self, nb_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256,nb_hidden)
        self.fc2 = nn.Linear(nb_hidden,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1,256)))
        x = self.fc2(x)
        return x


#class Net2(nn.Module):
    
############################################################################

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs = 25):
    eta = 1e-1
    criterion = nn.MSELoss()

    for e in trange(nb_epochs):
        acc_loss = 0

        #by mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

def compute_nb_errors(model, test_input, test_target, mini_batch_size):
    nb_errors = 0

    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        # output has mini_batch_size rows (number of samples) and 10 columns (output classes)
        _, preds = output.max(1)
        # so we take the arg maximum in each row (along dim 1)

        for k in range(mini_batch_size):
            if test_target[b+k, preds[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors


############################################################################
"""
# Question 2
mini_batch_size = 100
for k in range(10):
    model = Net(200)
    train_model(model, train_input, train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100*nb_test_errors)/test_input.size(0), nb_test_errors, test_input.size(0)))
"""

# Question 3
for h in [10,50,200,500,1000]:
    model = Net(h)
    train_model(model, train_input, train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net nh={:d} {:0.2f}%% {:d}/{:d}'.format(h, (100*nb_test_errors)/test_input.size(0), nb_test_errors, test_input.size(0)))







