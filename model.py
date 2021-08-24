import torch
import torch.nn as nn
import torch.nn.functional as nnf


class conv_block(nn.Module):

    def __init__(self, dim, in_channels, out_channels, mode='maintain'):

        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        # lightweight method choose.
        self.method = 'multi_conv'

        if mode == 'half':
            kernel_size = 3
            stride = 2
            padding = 1
            self.main = conv_fn(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        elif mode == 'maintain':
            kernel_size = 3
            stride = 1
            padding = 1
            self.main = conv_fn(in_channels, out_channels, kernel_size, stride, padding, bias=True)

        else:
            raise Exception('stride must be 1 or 2')
        self.act = nn.ReLU()

    def forward(self, out):
        """
        Pass the input through the conv_block
        """
        out = self.main(out)
        out = self.act(out)

        return out


class deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv, self).__init__()

        self.main = nn.ConvTranspose3d(in_channels, out_channels, 2, 2, bias=True)
        self.act = nn.ReLU()

    def forward(self, out):
        out = self.main(out)
        out = self.act(out)
        return out


class unet_model(nn.Module):
    ## 5/20
    def __init__(self, dim):
        super(unet_model, self).__init__()
        self.base_channel = 16
        self.l11 = conv_block(dim, 1, self.base_channel // 4)
        self.l12 = conv_block(dim, 1, self.base_channel // 4)
        self.l1 = conv_block(dim, 2, self.base_channel // 2)
        self.l1_ = conv_block(dim, self.base_channel, self.base_channel)  # base_channel

        self.l21 = conv_block(dim, self.base_channel // 4, self.base_channel // 2, 'half')
        self.l22 = conv_block(dim, self.base_channel // 4, self.base_channel // 2, 'half')
        self.l2 = conv_block(dim, self.base_channel, self.base_channel, 'half')
        self.l2_ = conv_block(dim, self.base_channel * 2, self.base_channel * 2)

        self.l31 = conv_block(dim, self.base_channel // 2, self.base_channel, 'half')
        self.l32 = conv_block(dim, self.base_channel // 2, self.base_channel, 'half')
        self.l3 = conv_block(dim, self.base_channel * 2, self.base_channel * 2, 'half')
        self.l3_ = conv_block(dim, self.base_channel * 4, self.base_channel * 4)

        self.l41 = conv_block(dim, self.base_channel, self.base_channel * 2, 'half')
        self.l42 = conv_block(dim, self.base_channel, self.base_channel * 2, 'half')
        self.l4 = conv_block(dim, self.base_channel * 4, self.base_channel * 4, 'half')
        self.l4_ = conv_block(dim, self.base_channel * 8, self.base_channel * 8)

        self.l51 = conv_block(dim, self.base_channel * 2, self.base_channel * 4, 'half')
        self.l52 = conv_block(dim, self.base_channel * 2, self.base_channel * 4, 'half')
        self.l5 = conv_block(dim, self.base_channel * 8, self.base_channel * 8, 'half')
        self.l5_ = conv_block(dim, self.base_channel * 16, self.base_channel * 8)

        self.d11 = conv_block(dim, self.base_channel * 16, self.base_channel * 8)
        self.d12 = conv_block(dim, self.base_channel * 8, self.base_channel * 4)

        self.d21 = conv_block(dim, self.base_channel * 8, self.base_channel * 4)
        self.d22 = conv_block(dim, self.base_channel * 4, self.base_channel * 2)

        self.d31 = conv_block(dim, self.base_channel * 4, self.base_channel * 4)
        self.d32 = conv_block(dim, self.base_channel * 4, self.base_channel * 2)

        self.d41 = conv_block(dim, 3 * self.base_channel, self.base_channel * 2)
        self.d42 = conv_block(dim, self.base_channel * 2, self.base_channel)
        # self.gen_flow = self.output(16, dim)

        self.d1 = deconv(self.base_channel * 8, self.base_channel * 8)
        self.d2 = deconv(self.base_channel * 4, self.base_channel * 4)
        self.d3 = deconv(self.base_channel * 2, self.base_channel * 2)
        self.d4 = deconv(self.base_channel * 2, self.base_channel * 2)

    def weight(self, inchannels, outchannels, k_size=1, stride=1, padding=0):
        layer = nn.Sequential(nn.Conv3d(inchannels, outchannels, k_size, stride, padding, bias=False),
                              nn.LeakyReLU())
        return layer

    def output(self, inchannels, outchannels, k_size=3, stride=1, padding=1):
        layer = nn.Sequential(nn.Conv3d(inchannels, outchannels, k_size, stride, padding, bias=False),
                              nn.Softsign())
        return layer

    def forward(self, x):
        x1 = x[:, 0, ...].unsqueeze(1)
        x2 = x[:, 1, ...].unsqueeze(1)

        x11 = self.l11(x1)
        x12 = self.l12(x2)
        x1_ = self.l1(x)
        x1 = self.l1_(torch.cat([x1_, x11, x12], 1))

        x21 = self.l21(x11)
        x22 = self.l22(x12)
        x2 = self.l2(x1)
        x2 = self.l2_(torch.cat([x2, x21, x22], 1))

        x31 = self.l31(x21)
        x32 = self.l32(x22)
        x3 = self.l3(x2)
        x3 = self.l3_(torch.cat([x3, x31, x32], 1))

        x41 = self.l41(x31)
        x42 = self.l42(x32)
        x4 = self.l4(x3)
        x4 = self.l4_(torch.cat([x4, x41, x42], 1))

        x51 = self.l51(x41)
        x52 = self.l52(x42)
        x5 = self.l5(x4)
        x5 = self.l5_(torch.cat([x5, x51, x52], 1))

        y = self.d1(x5)
        y = self.d11(torch.cat([y, x4], 1))
        y = self.d12(y)

        y = self.d2(y)
        y = self.d21(torch.cat([y, x3], 1))
        y = self.d22(y)

        y = self.d3(y)
        y = self.d31(torch.cat([y, x2], 1))
        y = self.d32(y)

        y = self.d4(y)
        y = self.d41(torch.cat([y, x1], 1))
        y = self.d42(y)

        return y


class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        self.unet_model = unet_model(dim)
        self.gen_flow_direction = nn.Conv3d(16, dim, kernel_size=3, padding=1, bias=False)
        self.gen_flow_inverse = nn.Conv3d(16, dim, kernel_size=3, padding=1, bias=False)

        self.norm = nn.Softsign()

    def forward(self, x):
        y = self.unet_model(x)

        flow_direction = self.gen_flow_direction(y)
        flow_inverse=self.gen_flow_inverse(y)
        flow_direction = self.norm(flow_direction)
        flow_inverse=self.norm(flow_inverse)

        return flow_direction,flow_inverse


class SpatialTransform(nn.Module):
    def __init__(self, ):
        super(SpatialTransform, self).__init__()

    def forward(self, x, flow, grid, mode='bilinear'):
        sample_grid = grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3]-0.5) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2]-0.5) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1]-0.5) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode=mode)

        return flow


class CompositionTransform(nn.Module):
    def __init__(self, ):
        super(CompositionTransform, self).__init__()

    def forward(self, flow_1, flow_2, grid, range_flow,mode='bilinear'):
        size_tensor = grid.size()
        sample_grid = grid + flow_1.permute(0,2,3,4,1)*range_flow
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3]-0.5) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2]-0.5) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1]-0.5) * 2
        compos_flow = torch.nn.functional.grid_sample(flow_2, sample_grid, mode=mode)+flow_1

        return compos_flow


class Auot_Folding_Adjustment(nn.Module):
    def __init__(self):
        super(Auot_Folding_Adjustment, self).__init__()
        self.input = nn.Sequential(nn.Conv3d(3, 8, 3, 1, 1, bias=True), nn.ReLU())
        self.l1 = nn.Sequential(nn.Conv3d(8, 16, 3, 1, 1, bias=True), nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv3d(16, 8, 3, 1, 1, bias=True), nn.ReLU())
        self.l3 = nn.Sequential(nn.Conv3d(8, 3, 3, 1, 1, bias=True))
        # self.norm = nn.Softsign()

    def forward(self, x):
        x1 = self.input(x)
        x = self.l1(x1)
        x = self.l2(x)
        x = self.l3(x1 + x)
        # flow = self.norm(x)
        return x
