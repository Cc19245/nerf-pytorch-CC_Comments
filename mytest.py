import numpy as np
import torch 

a = np.random.randn(6, 8, 3)
print(a)

b = a[..., :3]
print(b.shape)

print(id(b) == id(a))

c = a[:]
print(c.shape)
print(id(c) == id(a))

d = torch.randn(2, 3)
e = d[..., 0]
print(e.shape)