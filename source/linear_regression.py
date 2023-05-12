import torch
from torch import nn

class Linear_Regresssion_Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1)) # by default: requires_grad = True,dtype = torch.float32 
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        passes the input tensor through the computation graph while calculating
        the gradients of the necessary variables.
        returns: torch.Tensor
        """
        return (self.weights * x) + self.bias
    
