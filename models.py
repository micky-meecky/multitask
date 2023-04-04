from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)    # 3x3卷积核，padding=1，保证输入输出尺寸相同

# Conv3BN 是一个Conv3x3 + BN + ReLU的组合，这就是一个很常见的卷积块
class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)  # inplace=True，代表直接在原来的内存上修改，不再申请新的内存

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x

# UNetModule 是一个基本的卷积块，包含两个卷积层，每个卷积层后面跟一个BN层和一个ReLU激活函数
class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)  # 这里可以再添加一个参数，用来控制是否使用BN层，如：bn=True，默认是False。
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class PsiNet(nn.Module):
    """
    Adapted from Vanilla UNet implementation - https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample1 = nn.Upsample(scale_factor=2)
        upsample_bottom1 = nn.Upsample(scale_factor=bottom_s)
        upsample2 = nn.Upsample(scale_factor=2)
        upsample_bottom2 = nn.Upsample(scale_factor=bottom_s)
        upsample3 = nn.Upsample(scale_factor=2)
        upsample_bottom3 = nn.Upsample(scale_factor=bottom_s)

        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers1 = [upsample1] * len(self.up)
        self.upsamplers1[-1] = upsample_bottom1
        self.upsamplers2 = [upsample2] * len(self.up)
        self.upsamplers2[-1] = upsample_bottom2
        self.upsamplers3 = [upsample3] * len(self.up)
        self.upsamplers3[-1] = upsample_bottom3

        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final3 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        x_out1 = x_out
        x_out2 = x_out
        x_out3 = x_out

        # Decoder mask segmentation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers1, self.up))
        ):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))

        # Decoder contour estimation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers2, self.up))
        ):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))

        # Regression
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers3, self.up))
        ):
            x_out3 = upsample(x_out3)
            x_out3 = up(torch.cat([x_out3, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            if self.num_classes > 1:
                x_out2 = F.log_softmax(x_out2, dim=1)

        if self.add_output:
            x_out3 = self.conv_final3(x_out3)
            x_out3 = F.sigmoid(x_out3)

        return [x_out1, x_out2, x_out3]


class UNet_DCAN(nn.Module):
    """
    Adapted from Vanilla UNet implementation - https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample1 = nn.Upsample(scale_factor=2)
        upsample_bottom1 = nn.Upsample(scale_factor=bottom_s)
        upsample2 = nn.Upsample(scale_factor=2)
        upsample_bottom2 = nn.Upsample(scale_factor=bottom_s)

        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers1 = [upsample1] * len(self.up)
        self.upsamplers1[-1] = upsample_bottom1
        self.upsamplers2 = [upsample2] * len(self.up)
        self.upsamplers2[-1] = upsample_bottom2

        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        x_out1 = x_out
        x_out2 = x_out

        # Decoder mask segmentation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers1, self.up))
        ):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))

        # Decoder contour estimation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers2, self.up))
        ):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            if self.num_classes > 1:
                x_out2 = F.log_softmax(x_out2, dim=1)

        return [x_out1, x_out2]


class UNet_DMTN(nn.Module):
    """
    Adapted from Vanilla UNet implementation - https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels=3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample1 = nn.Upsample(scale_factor=2)
        upsample_bottom1 = nn.Upsample(scale_factor=bottom_s)
        upsample2 = nn.Upsample(scale_factor=2)
        upsample_bottom2 = nn.Upsample(scale_factor=bottom_s)

        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers1 = [upsample1] * len(self.up)
        self.upsamplers1[-1] = upsample_bottom1
        self.upsamplers2 = [upsample2] * len(self.up)
        self.upsamplers2[-1] = upsample_bottom2

        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        x_out1 = x_out
        x_out2 = x_out

        # Decoder mask segmentation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers1, self.up))
        ):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))

        # Regression
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers2, self.up))
        ):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            x_out2 = F.sigmoid(x_out2)

        return [x_out1, x_out2]


class UNet(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels=3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        padding=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        # assert 这里是为了保证输入的参数是正确的，如果不正确就会报错
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]

        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]

        # self.down, self.up 是 nn.ModuleList() 的实例，nn.ModuleList() 是一个容器，可以包含多个 nn.Module() 的实例,
        # 那么nn.module()是什么呢？nn.Module()是一个基类，所有的神经网络模块都继承自nn.Module()，所以nn.ModuleList()可以包含多个神经网络模块
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        # self.module(input_channels, down_filter_sizes[0]) 是一个 UNetModule() 的实例，UNetModule() 继承自 nn.Module()
        # down_filter_sizes[0]是第一个卷积层的输出通道数，input_channels是输入通道数,如果为down_filter_sizes[3]，那么就是第四个卷积层的输出通道数，即元祖中的第四个元素，是8.

        self.down.append(self.module(input_channels, down_filter_sizes[0]))  # nn.moduleList.append()是向容器中添加元素

        # prev_i 是前一个卷积层的索引，nf 是当前卷积层的输出通道数
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))    # 定义下采样的每一层的module，self.module()中的参数是上一层的输出通道数和当前层的输出通道数
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])    # + nf 是因为上采样后的特征图和下采样的特征图进行拼接
            )
        pool = nn.MaxPool2d(2, 2)   # 定义池化层, 第一个参数是池化核的大小，第二个参数是步长，填充默认为0
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        # [None]是一个列表，列表中只有一个元素，这个元素是None，[None] + [pool] * (len(self.down) - 1)是一个列表，列表中有len(self.down) - 1个元素，每个元素都是pool
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom   # 将最后一个元素赋值为pool_bottom，一共做了4次，第一次表示输入数据不需要进行下采样操作，第二次到第四次表示输入数据需要进行下采样操作
        self.upsamplers = [upsample] * len(self.up)  # [upsample] * len(self.up)是一个列表，列表中有len(self.up)个元素，每个元素都是upsample
        self.upsamplers[-1] = upsample_bottom   # 将最后一个元素赋值为upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, padding)   # 定义最后一层卷积层，输入通道数是up_filter_sizes[0]，输出通道数是num_classes，卷积核大小是padding

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)
            # print(x_out.shape)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers, self.up))
        ):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))
            # print(x_out.shape)

        if self.add_output:
            x_out = self.conv_final(x_out)
            # print(x_out.shape)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)

        return [x_out]


class UNet_ConvMCD(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final3 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers, self.up))
        ):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out)
            x_out2 = self.conv_final2(x_out)
            x_out3 = self.conv_final3(x_out)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)
                x_out2 = F.log_softmax(x_out2, dim=1)
            x_out3 = F.sigmoid(x_out3)

        # return x_out,x_out1,x_out2,x_out3
        return [x_out1, x_out2, x_out3]