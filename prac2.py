import torch
from torch import Tensor
import dlc_practical_prologue as prologue

# 1. Nearest neighbor
def nearest_classification(train_input, train_target, x):
    dist = (train_input - x).pow(2).sum(1).view(-1)
    _, n = torch.min(dist, 0)
    return train_target[n.item()]

# 2. Error estimation
def compute_nb_errors(train_input, train_target, test_input,
                      test_target, mean=None, proj=None):
    if mean is not None:
       train_input -= mean
       test_input -= mean
    if proj is not None:
        train_input = train_input @ proj.t()
        test_input = test_input @ proj.t()

    nb_errors = 0
    
    for n in range(test_input.size(0)):
        y = nearest_classification(train_input, train_target,
                                   test_input[n])
        if y != test_target[n]:
            nb_errors += 1

    return nb_errors


# 3. 
def PCA(x):
    mean = x.mean(0)
    x0 = x - mean
    S = x0.t() @ x0
    eigen_val, eigen_vec = S.eig(True)
    eigen_valmod = torch.empty(eigen_val.shape[0],1)
    for i in range(len(eigen_val)):
        eigen_valmod[i] = torch.linalg.norm(eigen_val[i,:],2)
    order = eigen_valmod().sort(descending=True)[1]
    eigen_vec = eigen_vec[:,order]
    return mean, eigen_vec


 
