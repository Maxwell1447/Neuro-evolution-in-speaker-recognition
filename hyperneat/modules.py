import torch
import torch.nn as nn
from torch.nn.functional import conv1d, conv2d


class Conv1DCppnVar(torch.nn.Module):

    def __init__(self, C_in, C_out, kernel_size, cppn, input_size, z=0, activation=torch.relu):
        super().__init__()

        self.input_size = input_size
        self.k = kernel_size
        self.C_in = C_in
        self.C_out = C_out

        self.activation = activation

        c_out = torch.arange(C_out)
        c_in = torch.arange(C_in)
        out = torch.arange(self.output_size)
        in_ = torch.arange(self.k)

        cc_out, cc_in, oo, kk = torch.meshgrid([c_out, c_in, out, in_])
        ii = kk + torch.arange(self.output_size).expand(self.k, self.output_size).t()

        self.w = cppn[1](x_in=ii, x_out=oo,
                         C_in=cc_in, C_out=cc_out,
                         z=torch.full(ii.shape, z))
        self.b = cppn[0](x_in=torch.zeros_like(c_out), x_out=torch.zeros_like(c_out),
                         C_in=torch.zeros_like(c_out), C_out=c_out,
                         z=torch.full(c_out.shape, z))
        self.w.requires_grad = False
        self.b.requires_grad = False

    @property
    def output_size(self):
        return self.input_size - self.k

    def forward(self, x):
        # x [N, C_in, input_size]
        assert x.shape[1] == self.C_in
        assert x.shape[2] == self.input_size

        idxs = torch.arange(self.k) + torch.arange(self.output_size).expand(self.k, self.output_size).t()
        return self.activation((x.expand(self.C_out, x.shape[0], self.C_in, self.input_size)
                                .transpose(0, 1)[:, :, :, idxs]
                                * self.w).sum(dim=[2, 4]) + self.b.expand(self.output_size, self.C_out).t())


class Conv1DCppn(torch.nn.Module):

    def __init__(self, C_in, C_out, kernel_size, cppn, z=0, activation=torch.relu, padding=0):
        super().__init__()

        self.k = kernel_size
        self.C_in = C_in
        self.C_out = C_out
        self.activation = activation
        self.padding = padding

        c_out = torch.arange(C_out)
        c_in = torch.arange(C_in)
        k = torch.arange(kernel_size)

        cc_out, cc_in, kk = torch.meshgrid([c_out, c_in, k])

        self.w = cppn[0](k=kk,
                         C_in=cc_in, C_out=cc_out,
                         z=torch.full(kk.shape, z))

        self.b = cppn[1](k=torch.zeros_like(c_out),
                         C_in=torch.zeros_like(c_out), C_out=c_out,
                         z=torch.full(c_out.shape, z))

    def forward(self, x):
        # x [N, C_in, input_size]
        assert x.shape[1] == self.C_in, "{} != {}".format(x.shape[1], self.C_in)

        return self.activation(conv1d(x, self.w, bias=self.b, padding=self.padding))


class AudioCNN(torch.nn.Module):

    def __init__(self, cppn):
        super().__init__()

        self.conv1 = Conv1DCppn(1, 16, 20, cppn, z=0, activation=torch.relu)
        self.conv2 = Conv1DCppn(16, 24, 15, cppn, z=1, activation=torch.relu)
        self.conv3 = Conv1DCppn(24, 32, 10, cppn, z=2, activation=torch.relu)
        self.conv4 = Conv1DCppn(32, 64, 5, cppn, z=3, activation=torch.relu)
        self.conv5 = Conv1DCppn(64, 72, 5, cppn, z=4, activation=torch.relu)

        self.bn1 = nn.BatchNorm1d(16, affine=False)
        self.bn2 = nn.BatchNorm1d(24, affine=False)
        self.bn3 = nn.BatchNorm1d(32, affine=False)
        self.bn4 = nn.BatchNorm1d(64, affine=False)
        self.bn5 = nn.BatchNorm1d(72, affine=False)

        self.mp1 = nn.MaxPool1d(10, stride=10)
        self.mp2 = nn.MaxPool1d(10, stride=10)
        self.mp3 = nn.MaxPool1d(7, stride=7)
        self.mp4 = nn.MaxPool1d(6, stride=6)
        self.mp5 = nn.MaxPool1d(4, stride=4)

        self.l1 = nn.Linear(72, 1)

    def forward(self, x):

        x = self.mp1(self.bn1(self.conv1(x)))
        x = self.mp2(self.bn2(self.conv2(x)))
        x = self.mp3(self.bn3(self.conv3(x)))
        x = self.mp4(self.bn4(self.conv4(x)))
        x = self.mp5(self.bn5(self.conv5(x)))

        return x


class AudioCNNClassical(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, 20)
        self.conv2 = nn.Conv1d(16, 24, 15)
        self.conv3 = nn.Conv1d(24, 32, 10)
        self.conv4 = nn.Conv1d(32, 64, 5)
        self.conv5 = nn.Conv1d(64, 72, 5)

        self.bn1 = nn.BatchNorm1d(16, affine=False)
        self.bn2 = nn.BatchNorm1d(24, affine=False)
        self.bn3 = nn.BatchNorm1d(32, affine=False)
        self.bn4 = nn.BatchNorm1d(64, affine=False)
        self.bn5 = nn.BatchNorm1d(72, affine=False)

        self.mp1 = nn.MaxPool1d(10, stride=10)
        self.mp2 = nn.MaxPool1d(10, stride=10)
        self.mp3 = nn.MaxPool1d(7, stride=7)
        self.mp4 = nn.MaxPool1d(6, stride=6)
        self.mp5 = nn.MaxPool1d(4, stride=4)

        self.l1 = nn.Linear(72, 1)

    def forward(self, x):

        x = self.mp1(self.bn1(torch.relu(self.conv1(x))))
        x = self.mp2(self.bn2(torch.relu(self.conv2(x))))
        x = self.mp3(self.bn3(torch.relu(self.conv3(x))))
        x = self.mp4(self.bn4(torch.relu(self.conv4(x))))
        x = self.mp5(self.bn5(torch.relu(self.conv5(x))))
        x = torch.sigmoid(self.l1(x.view(-1, 72)))
        return x


class Conv2DCppn(torch.nn.Module):

    def __init__(self, C_in, C_out, kernel_size: int, cppn, z=0, activation=torch.relu, padding=0, device=torch.device("cpu")):
        super().__init__()

        kernel_size = (kernel_size, kernel_size)
        self.k_x = kernel_size[0]
        self.k_y = kernel_size[1]
        self.C_in = C_in
        self.C_out = C_out
        self.activation = activation
        self.padding = padding

        c_out = torch.arange(C_out)
        c_in = torch.arange(C_in)
        kx = torch.arange(self.k_x)
        ky = torch.arange(self.k_y)

        cc_out, cc_in, kk_x, kk_y = torch.meshgrid([c_out, c_in, kx, ky])

        self.w = cppn[0](k_x=kk_x, k_y=kk_y,
                         C_in=cc_in, C_out=cc_out,
                         x_in=torch.zeros_like(kk_x), x_out=torch.zeros_like(kk_x),
                         z=torch.full(kk_x.shape, z)).to(device)

        self.b = cppn[1](k_x=torch.zeros_like(c_out), k_y=torch.zeros_like(c_out),
                         C_in=torch.zeros_like(c_out), C_out=c_out,
                         x_in=torch.zeros_like(c_out), x_out=torch.zeros_like(c_out),
                         z=torch.full(c_out.shape, z)).to(device)

    def forward(self, x):
        # x [N, C_in, input_size]
        assert x.shape[1] == self.C_in, "{} != {}".format(x.shape[1], self.C_in)

        return self.activation(conv2d(x, self.w, bias=self.b, padding=self.padding))


class LinearFromConv2DCppn(torch.nn.Module):

    def __init__(self, input_size, output_size, cppn, z=0, activation=torch.sigmoid, padding=0, device=torch.device("cpu")):
        super().__init__()

        self.activation = activation

        x_in = torch.arange(input_size)
        x_out = torch.arange(output_size)

        xx_in, xx_out = torch.meshgrid([x_in, x_out])

        self.w = cppn[0](k_x=torch.zeros_like(xx_in), k_y=torch.zeros_like(xx_in),
                         C_in=torch.zeros_like(xx_in), C_out=torch.zeros_like(xx_in),
                         x_in=xx_in, x_out=xx_out,
                         z=torch.full(xx_in.shape, z)).to(device)

        self.b = cppn[1](k_x=torch.zeros_like(x_out), k_y=torch.zeros_like(x_out),
                         C_in=torch.zeros_like(x_out), C_out=torch.zeros_like(x_out),
                         x_in=torch.zeros_like(x_out), x_out=x_out,
                         z=torch.full(x_out.shape, z)).to(device)

    def forward(self, x):
        # x [N, input_size]

        return self.activation(torch.matmul(x, self.w) + self.b)


class LeNet5Cppn(torch.nn.Module):

    def __init__(self, cppn, device=torch.device("cpu")):
        super().__init__()

        self.conv1 = Conv2DCppn(1, 16, 5, cppn, z=0, device=device)
        self.conv2 = Conv2DCppn(16, 24, 4, cppn, z=1, device=device)
        self.conv3 = Conv2DCppn(24, 32, 2, cppn, z=2, device=device)

        self.bn1 = nn.BatchNorm2d(16, affine=False)
        self.bn2 = nn.BatchNorm2d(24, affine=False)
        self.bn3 = nn.BatchNorm2d(32, affine=False)

        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.mp3 = nn.MaxPool2d(2, stride=2)

        self.l1 = LinearFromConv2DCppn(32, 10, cppn=cppn, z=5, device=device)

    def forward(self, x):

        x = self.mp1(self.bn1(self.conv1(x)))
        x = self.mp2(self.bn2(self.conv2(x)))
        x = self.mp3(self.bn3(self.conv3(x)))

        x = torch.sigmoid(self.l1(x.view(-1, 32)))

        return x