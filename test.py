import torch 
from torch import nn
score = torch.tensor([[100,0,0,0,0,0]],dtype=torch.float32)
# target = torch.zeros(score.shape[0],device=score.device, dtype=torch.long)
target = torch.zeros((1,),dtype=torch.long)
print(target)
loss_fcn = nn.CrossEntropyLoss()

print(loss_fcn(score,target))