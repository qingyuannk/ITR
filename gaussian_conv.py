# 实现二维高斯卷积，卷积核大小=[7, 7]，sigma=[2, 2]，stride=1。输入二维矩阵大小[11, 11]，数值随机。
# 要求：1）输出二维矩阵大小和输入二维矩阵大小一致；2）border padding的方式为reflect。

import torch
import torchvision
from torch import nn


# 输入
INPUT_SIZE = [11, 11]
KERNEL_SIZE = [7, 7]
SIGMA = [2, 2]
STRIDE = 1


# PyTorch 实现
torch_gaussian_conv = torchvision.transforms.GaussianBlur(KERNEL_SIZE, SIGMA)


# 手工实现
class MyGaussianConv(nn.Module):
    def __init__(self, kernel_size, sigma, stride):
        super().__init__()
        self.stride = stride
        kernel1d_x = self._get_gaussian_kernel1d(kernel_size[0], sigma[0])
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size[1], sigma[1])
        self.kernel = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])

    @staticmethod
    def _get_gaussian_kernel1d(kernel_size: int, sigma: float):
        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d

    def conv2d(self, input):
        input_h, input_w = input.shape
        kernel_h, kernel_w = self.kernel.shape

        output_h = (input_h - kernel_h) // self.stride + 1
        output_w = (input_w - kernel_w) // self.stride + 1
        output = torch.zeros(output_h, output_w)
        for i in range(0, input_h - kernel_h + 1, self.stride):
            for j in range(0, input_w - kernel_w + 1, self.stride):
                region = input[i:i+kernel_h, j:j+kernel_w]
                output[i // self.stride, j // self.stride] = torch.sum(region * self.kernel)

        return output

    def pad(self, input):
        pad_h = self.kernel.shape[0] // 2
        pad_w = self.kernel.shape[1] // 2
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

        return padded

    def forward(self, input):
        # TODO: generalize to batch and channel dimension
        input = input.squeeze()
        input = self.pad(input)
        input = self.conv2d(input)
        return input


my_gaussian_conv = MyGaussianConv(KERNEL_SIZE, SIGMA, STRIDE)


# 测试
def test(input):
    EPS = 1e-6
    torch_output = torch_gaussian_conv(input).squeeze()
    my_output = my_gaussian_conv(input)

    if torch_output.shape != my_output.shape:
        print(f'the output shape should be {torch_output.shape} instead of {my_output.shape}')
        return False

    diff = torch.flatten(torch_output) - torch.flatten(my_output)
    for x in diff:
        if x.abs() > EPS:
            return False

    return True


num_test = 5
for i in range(1, num_test + 1):
    input = torch.rand(INPUT_SIZE).reshape(1, 1, INPUT_SIZE[0], INPUT_SIZE[1])
    print(f'pass test #{i}: {test(input)}')
