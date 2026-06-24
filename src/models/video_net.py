import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from ..layers.layers import subpel_conv1x1, conv3x3, DepthConvBlock, DepthConvBlock2

backward_grid = [{} for _ in range(9)]  # 0~7 for GPU, -1 for CPU


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


# pylint: enable=W0221

def add_grid_cache(flow):
    device_id = -1 if flow.device == torch.device('cpu') else flow.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=torch.float32).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=torch.float32).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    add_grid_cache(flow)
    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)

    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature

class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = int(dim / n_div)
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = nn.Conv2d(dim, ouc, 1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x
    

class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, end_with_relu=False,
                 bottleneck=False, inplace=False):
        super().__init__()
        in_channel = channel // 2 if bottleneck else channel
        self.first_layer = nn.LeakyReLU(negative_slope=slope, inplace=False)
        self.conv1 = nn.Conv2d(channel, in_channel, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.conv2 = nn.Conv2d(in_channel, channel, 3, padding=1)
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return identity + out
    

class ResBlock_PConv(nn.Module):
    def __init__(self, channel, slope=0.01, end_with_relu=False,
                 bottleneck=False, inplace=False, n_div=4, forward='split_cat'):
        super(ResBlock_PConv, self).__init__()
        in_channel = channel // 2 if bottleneck else channel
        self.first_layer = nn.LeakyReLU(negative_slope=slope, inplace=False)
        self.conv1 = PConv(channel, in_channel, n_div=n_div, forward=forward)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.conv2 = PConv(in_channel, channel, n_div=n_div, forward=forward)
        self.last_layer = self.relu if end_with_relu else nn.Identity()
        
    def forward(self, x):
        identity = x
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return identity + out
    
from functools import partial
def get_block_fn(inplace, blok_type='normal', n_div=4):
    if blok_type == 'normal':
        return partial(ResBlock, inplace=inplace)
    elif blok_type == 'pconv':
        return partial(ResBlock_PConv, inplace=inplace, n_div=n_div)
    else:
        raise ValueError(f"Unsupported blok_type: {blok_type}")


class MEBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for _ in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                   self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                      flow_warp(im2_list[img_index], flow_up),
                                                      flow_up], 1))

        return flow
    
class GrayBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(4, 32, 7, 1, padding=3)  # 修改为 4 通道输入
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class Gray_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([GrayBasic() for _ in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)

        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                   self.moduleBasic[level](torch.cat([
                       im1_list[img_index],
                       flow_warp(im2_list[img_index], flow_up),
                       flow_up
                   ], dim=1))

        return flow
    
    
class GrayBasic_FM(nn.Module):
    def __init__(self, complexity_level=0):
        super().__init__()
        self.relu = nn.ReLU()
        self.by_pass = False
        if complexity_level < 0:
            self.by_pass = True
        elif complexity_level == 0:
            self.conv1 = nn.Conv2d(4, 32, 7, 1, padding=3)
            self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
            self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
            self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
            self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        elif complexity_level == 3:
            self.conv1 = nn.Conv2d(4, 32, 5, 1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
            self.conv3 = nn.Conv2d(64, 32, 5, 1, padding=2)
            self.conv4 = nn.Conv2d(32, 16, 5, 1, padding=2)
            self.conv5 = nn.Conv2d(16, 2, 5, 1, padding=2)

    def forward(self, x):
        if self.by_pass:
            return x[:, -2:, :, :]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class Gray_Spynet_FM(nn.Module):
    def __init__(self):
        super().__init__()
        self.me_8x = GrayBasic_FM(0)
        self.me_4x = GrayBasic_FM(0)
        self.me_2x = GrayBasic_FM(3)
        self.me_1x = GrayBasic_FM(3)

    def forward(self, im1, im2):
        batchsize = im1.size()[0]

        im1_1x = im1
        im1_2x = F.avg_pool2d(im1_1x, kernel_size=2, stride=2)
        im1_4x = F.avg_pool2d(im1_2x, kernel_size=2, stride=2)
        im1_8x = F.avg_pool2d(im1_4x, kernel_size=2, stride=2)
        im2_1x = im2
        im2_2x = F.avg_pool2d(im2_1x, kernel_size=2, stride=2)
        im2_4x = F.avg_pool2d(im2_2x, kernel_size=2, stride=2)
        im2_8x = F.avg_pool2d(im2_4x, kernel_size=2, stride=2)

        shape_fine = im1_8x.size()
        zero_shape = [batchsize, 2, shape_fine[2], shape_fine[3]]
        flow_8x = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        flow_8x = self.me_8x(torch.cat((im1_8x, im2_8x, flow_8x), dim=1))

        flow_4x = bilinearupsacling(flow_8x) * 2.0
        flow_4x = flow_4x + self.me_4x(torch.cat((im1_4x,
                                                  flow_warp(im2_4x, flow_4x),
                                                  flow_4x),
                                                 dim=1))

        flow_2x = bilinearupsacling(flow_4x) * 2.0
        flow_2x = flow_2x + self.me_2x(torch.cat((im1_2x,
                                                  flow_warp(im2_2x, flow_2x),
                                                  flow_2x),
                                                 dim=1))

        flow_1x = bilinearupsacling(flow_2x) * 2.0
        flow_1x = flow_1x + self.me_1x(torch.cat((im1_1x,
                                                  flow_warp(im2_1x, flow_1x),
                                                  flow_1x),
                                                 dim=1))
        return flow_1x



class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class UNet2(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock2(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock2(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock2(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock2(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock2(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


def get_hyper_enc_dec_models(y_channel, z_channel, reduce_enc_layer=False, inplace=False):
    if reduce_enc_layer:
        enc = nn.Sequential(
            nn.Conv2d(y_channel, z_channel, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )
    else:
        enc = nn.Sequential(
            conv3x3(y_channel, z_channel),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel, stride=2),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel, stride=2),
        )

    dec = nn.Sequential(
        conv3x3(z_channel, y_channel),
        nn.LeakyReLU(inplace=inplace),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(inplace=inplace),
        conv3x3(y_channel, y_channel),
        nn.LeakyReLU(inplace=inplace),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(inplace=inplace),
        conv3x3(y_channel, y_channel),
    )

    return enc, dec


class freup_Areadinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Areadinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H / 2), 0:int(W / 2)] = output[:, :, 0:int(H / 2), 0:int(W / 2)]
        crop[:, :, int(H / 2):H, 0:int(W / 2)] = output[:, :, int(H * 1.5):2 * H, 0:int(W / 2)]
        crop[:, :, 0:int(H / 2), int(W / 2):W] = output[:, :, 0:int(H / 2), int(W * 1.5):2 * W]
        crop[:, :, int(H / 2):H, int(W / 2):W] = output[:, :, int(H * 1.5):2 * H, int(W * 1.5):2 * W]
        crop = F.interpolate(crop, (2 * H, 2 * W))

        return self.post(crop)


class fresadd(nn.Module):
    def __init__(self, channels=32):
        super(fresadd, self).__init__()

        self.Fup = freup_Areadinterpolation(channels)

        self.fuse = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x

        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')

        x3 = self.Fup(x1)

        xm = x2 + x3
        xn = self.fuse(xm)

        return xn
