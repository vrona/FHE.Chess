
import torch
import numpy as np

x = torch.empty(32,2,8,8)

y = x[:,0,:]
z = x[:,1,:]



print(x.shape, y.shape)
print(y)

#print(y,'\n',y.shape)

