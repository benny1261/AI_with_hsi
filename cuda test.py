import torch

x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")