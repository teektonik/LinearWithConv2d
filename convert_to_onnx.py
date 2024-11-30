import torch
import random

from layers import LinearWithConv2d, Linear_


lower = 1
upper = 10

in_features = random.randint(lower, upper)
out_features = random.randint(lower, upper)
batch_size = random.randint(lower, upper)

torch_input = torch.rand(batch_size, in_features)

torch_model = Linear_(in_features, out_features, bias=True)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("linear.onnx")

torch_model = LinearWithConv2d(in_features, out_features, bias=True)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("linear_conv.onnx")
