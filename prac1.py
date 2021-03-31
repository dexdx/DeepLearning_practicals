import torch
import time

# 1. Multiple views of a storage
print("1. Multiple views of a storage")
m = torch.full((13,13),1)
m[:,[1,6,11]] = 2
m[[1,6,11],:] = 2
m[3:5,3:5] = 3
m[3:5,8:10] = 3
m[8:10,3:5] = 3
m[8:10,8:10] = 3
print(m,'\n')

# 2. Eigendecomposition
M = torch.empty((20,20))
M.normal_(0,1)
M_inv = M.inverse()
D = torch.diag(torch.arange(1.,21.))
A = M_inv.mm(D).mm(M)
e, _  = torch.eig(A)
print("2. Eigenvalues:",e[:,0].sort()[0], sep='\n')

# 3. Flops per second
a = torch.empty((5000,5000)).normal_(0,1)
b = torch.empty((5000,5000)).normal_(0,1)

t = time.perf_counter()
torch.mm(a,b)
t = time.perf_counter() - t
print('\n3. Flops per second')
print('time:',t,'seconds')
print(5000**3/t, 'floating point products per second')


# 4. Playing with strides
def mul_row(m):
    M = torch.empty((m.size(0),m.size(1)))
    n = M.size(0)
    for i in range(n):
        M[i,:] = m[i,:] * (i+1)
    return M

def mul_row_fast(m):
    n = m.size()[0]
    b = torch.diag(torch.arange(1,n+1).float())
    return torch.mm(b,m)

m = torch.empty((1000,400)).normal_()
t1 = time.perf_counter()
a = mul_row(m)
t2 = time.perf_counter()
b = mul_row_fast(m)
t3 = time.perf_counter()

print('\n4. Playing with strides')
print('slow/fast:', (t2-t1)/(t3-t2))
print('(Sanity check) difference between outputs:', torch.linalg.norm(a-b,'fro').item())
