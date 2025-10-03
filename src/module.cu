#include <torch/extension.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b);
torch::Tensor mul(torch::Tensor a, torch::Tensor b);
torch::Tensor gelu(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("add",&add,"speedy add");
    m.def("mul",&mul,"speedy mul");
    m.def("gelu",&gelu,"speedy gelu activation");
}