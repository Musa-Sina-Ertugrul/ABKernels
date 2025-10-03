#include <torch/extension.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b);
torch::Tensor mul(torch::Tensor a, torch::Tensor b);
torch::Tensor gelu(torch::Tensor input);
torch::Tensor gelu_backward(torch::Tensor input);
torch::Tensor sigmoid(torch::Tensor input);
torch::Tensor sigmoid_backward(torch::Tensor input);
torch::Tensor silu(torch::Tensor input);
torch::Tensor silu_backward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("add",&add,"speedy add");
    m.def("mul",&mul,"speedy mul");
    m.def("gelu",&gelu,"speedy gelu activation");
    m.def("gelu_backward",&gelu_backward,"speedy gelu activation");
    m.def("sigmoid",&sigmoid,"speedy sigmoid activation");
    m.def("sigmoid_backward",&sigmoid_backward,"speedy sigmoid activation");
    m.def("silu",&silu,"speedy silu activation");
    m.def("silu_backward",&silu_backward,"speedy silu activation");
}