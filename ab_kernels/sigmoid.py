import timeit
import torch
import torch.nn.functional as F
from torch.autograd import Function
import ab_kernels_cuda

class _Sigmoid(Function):
  
    @staticmethod
    def forward(ctx,input:torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return ab_kernels_cuda.sigmoid(input)

    @staticmethod
    def backward(ctx,grad_output:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        input = ctx.saved_tensors
        return ab_kernels_cuda.mul(grad_output,ab_kernels_cuda.sigmoid_backward(input))
    
def sigmoid(input:torch.Tensor) -> torch.Tensor:
    return _Sigmoid.apply(input)

if __name__ == "__main__":

    x_1 = torch.ones((160000,),dtype=torch.bfloat16,device='cuda')
    #print(ab_kernels_cuda.add(x_1,x_2))
    s = timeit.default_timer()
    sigmoid(x_1)
    e = timeit.default_timer()
    print("TIME: ",str(e-s))
    s = timeit.default_timer()
    F.sigmoid(x_1)
    e = timeit.default_timer()
    print("TIME: ",str(e-s))