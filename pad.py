import torch
from torch import nn

torch_reflection_pad = nn.ReflectionPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
padded1 = torch_reflection_pad(input)
print(padded1)

input = input.squeeze()
pad_h = 2
pad_w = 2
padded = torch.zeros(input.shape[-1] + pad_h * 2, input.shape[-2] + pad_w * 2)
# center
padded[pad_h:-pad_h, pad_w:-pad_w] = input
# pad edge
padded[:pad_h, pad_w:-pad_w] = input[1:1+pad_h].flip(dims=[0])
padded[-pad_h:, pad_w:-pad_w] = input[-pad_h-1:-1].flip(dims=[0])
padded[pad_h:-pad_h, :pad_w] = input[:, 1:1+pad_w].flip(dims=[1])
padded[pad_h:-pad_h, -pad_w:] = input[:, -pad_w-1:-1].flip(dims=[1])
# pad corner
padded[:pad_h, :pad_w] = input[1:1+pad_h, 1:1+pad_w].flip(dims=[0, 1])
padded[:pad_h, -pad_w:] = input[1:1+pad_h, -pad_w-1:-1].flip(dims=[0, 1])
padded[-pad_h:, :pad_w] = input[-pad_h-1:-1, 1:1+pad_w].flip(dims=[0, 1])
padded[-pad_h:, -pad_w:] = input[-pad_h-1:-1, -pad_w-1:-1].flip(dims=[0, 1])

assert torch.sum(padded - padded1) == 0
