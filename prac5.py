import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from math import pi

#pi = torch.acos(torch.zeros(1)).item() * 2

# Question 1
def generate_disc_set(nb):
    x = torch.empty((nb,2), dtype=torch.float32).uniform_(-1,1)
    rad = torch.tensor((2./pi)).sqrt().item()
    """
    y = torch.empty((nb), dtype=torch.int64)
    for i in range(nb):
        if x[i].norm() < rad:
            y[i] = 1
        else:
            y[i] = 0
    """
    # smarter way of creating y
    y = x.pow(2).sum(1).sub(2/pi).sign().add(1).div(2).long() 
    return x, y

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

# normalize sets
mu, std = train_input.mean(0), train_input.std(0)
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)

# Question 2
def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    mini_batch_size = 100
    nb_epochs = 250

    for e in trange(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_nb_errors(model, data_input, data_target):
    mini_batch_size = 100
    nb_errors = 0
    pred = model(data_input).max(1)[1]
    for b in range(0, data_input.size(0), mini_batch_size):
         nb_errors += (pred - data_target).abs().sum()
    return nb_errors


# Question 3
class create_shallow_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

create_deep_model = nn.Sequential(
    nn.Linear(2,4), nn.ReLU(),
    nn.Linear(4,8), nn.ReLU(),
    nn.Linear(8,16), nn.ReLU(),
    nn.Linear(16,32), nn.ReLU(),
    nn.Linear(32,64), nn.ReLU(),
    nn.Linear(64,128), nn.ReLU(),
    nn.Linear(128,2)
)


# Question 4
shallow = create_shallow_model()

# PyTorch default parameter initialization
print('Shallow net - default init')
train_model(shallow, train_input, train_target)
print('train error: {:0.2f}%, test error: {:0.2f}%\n'.format(100*compute_nb_errors(shallow,train_input,train_target)/train_input.size(0), 100*compute_nb_errors(shallow, test_input, test_target)/test_input.size(0)))

# Manual initialization (normal)
shallow = create_shallow_model()
sds = torch.tensor([1e-3,1e-2,1e-1,1.])

print('Shallow net - manual init')
for s in range(len(sds)):
    for p in shallow.parameters():
        p.data.normal_(0,sds[s])
    
    train_model(shallow, train_input, train_target)
    print('sd: {:e} || train error: {:0.2f}%, test error: {:0.2f}%'.format(sds[s], 100*compute_nb_errors(shallow,train_input,train_target)/train_input.size(0), 100*compute_nb_errors(shallow, test_input, test_target)/test_input.size(0)))


deep = create_deep_model
