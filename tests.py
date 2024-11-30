import pytest
import torch

from layers import LinearWithConv2d, Linear_


@pytest.fixture(scope="function", params=[
(3, 4, 1),
(34, 19, 52),
(5, 2, 3)
])

def param_test(request):
    return request.param

def test_func(param_test):
    (in_features, out_features, batch_size) = param_test
    linear_with_conv = LinearWithConv2d(in_features, out_features, bias=True)
    linear = Linear_(in_features, out_features, bias=True)

    with torch.no_grad():
        linear_with_conv.conv.weight.copy_(linear.linear.weight.unsqueeze(-1).unsqueeze(-1))
        if linear.linear.bias is not None:
            linear_with_conv.conv.bias.copy_(linear.linear.bias)
    
    x = torch.rand(batch_size, in_features)
    output_linear = linear(x)
    output_conv = linear_with_conv(x)
    diff = (output_linear - output_conv).abs().max().item()
    
    print("in_features: {0}, out_features: {1}, batch_size: {2}, difference: {3}".format(in_features, out_features, batch_size, diff))
    assert diff<1e-6