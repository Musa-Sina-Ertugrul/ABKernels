import timeit
import torch
import torch.nn.functional as F
from torch.autograd import Function
from load import _kernel

class _Gelu(Function):
  
    @staticmethod
    def forward(ctx,input:torch.Tensor) -> torch.Tensor:
        return _kernel.gelu(input)

    @staticmethod
    def backward(ctx,grad_output:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        return _kernel.gelu_backward(grad_output)
    
def gelu(input:torch.Tensor) -> torch.Tensor:
    return _Gelu.apply(input)

if __name__ == "__main__":

    x_1 = torch.ones((160000,),dtype=torch.bfloat16,device='cuda')
    #print(_kernel.add(x_1,x_2))
    s = timeit.default_timer()
    gelu(x_1)
    e = timeit.default_timer()
    print("TIME: ",str(e-s))
    s = timeit.default_timer()
    F.gelu(x_1)
    e = timeit.default_timer()
    print("TIME: ",str(e-s))