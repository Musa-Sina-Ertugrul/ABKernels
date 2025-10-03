import timeit
import torch
from torch.autograd import Function
import ab_kernels_cuda

class _Add(Function):
  
    @staticmethod
    def forward(ctx,left:torch.Tensor,right:torch.Tensor) -> torch.Tensor:
        return ab_kernels_cuda.add(left,right)

    @staticmethod
    def backward(ctx,grad_output:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        return grad_output,grad_output
    
def add(left:torch.Tensor,right:torch.Tensor) -> torch.Tensor:
    return _Add.apply(left,right)

if __name__ == "__main__":

    x_1 = torch.ones((160000,),dtype=torch.bfloat16,device='cuda')
    x_2 = torch.ones((160000,),dtype=torch.bfloat16,device='cuda')
    #print(ab_kernels_cuda.add(x_1,x_2))
    s = timeit.default_timer()
    add(x_1,x_2)
    e = timeit.default_timer()
    print("TIME: ",str(e-s))
    s = timeit.default_timer()
    torch.add(x_1,x_2)
    e = timeit.default_timer()
    print("TIME: ",str(e-s))