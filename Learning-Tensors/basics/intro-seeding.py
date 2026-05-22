import torch

# introduce seeding

torch.manual_seed(42)

# random tensor
r1 = torch.rand(2, 2)
print(r1)

# random tensor 2
r2 = torch.rand(2, 2)
print(r2)

# using same seed, should match r1
torch.manual_seed(42)
r3 = torch.rand(2, 2)
print(r3)
