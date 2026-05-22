import torch

# rank 2 tensor with only zeros
z = torch.zeros(5, 3)
print(z)
print(z.dtype)

# rank 2 tensor with only 1s and 16 in

i = torch.ones((5, 3), dtype=torch.int16)
print(i)
