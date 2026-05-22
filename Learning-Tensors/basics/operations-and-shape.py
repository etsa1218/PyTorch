import torch

# basic rank 2 tensor of only 1

ones = torch.ones(2, 3)
print(ones)

# multiplication by a scalar

twos = torch.ones(2, 3) * 2
print(twos)

# tensor addition
threes = ones + twos
print(threes)
print(threes.shape)

# will not work herre due to tensors not matching shape
r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
r3 = r1 + r2
