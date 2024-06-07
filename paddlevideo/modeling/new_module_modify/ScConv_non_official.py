'''
Description:
Date: 2023-07-21 14:36:27
LastEditTime: 2023-07-27 18:41:47
FilePath: /chengdongzhou/ScConv.py

https://zhuanlan.zhihu.com/p/649680775
https://github.com/cheng-haha/ScConv/blob/main/ScConv.py

https://www.elecfans.com/d/2245908.html
'''
'''
Description: 
Date: 2023-07-21 14:36:27
LastEditTime: 2023-07-27 18:41:47
FilePath: /chengdongzhou/ScConv.py
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GroupBatchnorm2d(nn.Layer):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = self.create_parameter(shape=[c_num, 1, 1], default_initializer=nn.initializer.Normal())
        self.bias = self.create_parameter(shape=[c_num, 1, 1], default_initializer=nn.initializer.Constant(value=0.0))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape([N, self.group_num, -1])
        mean = paddle.mean(x, axis=2, keepdim=True)
        std = paddle.std(x, axis=2, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        x = x.reshape([N, C, H, W])
        return x * self.weight + self.bias


class SRU(nn.Layer):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_threshold: float = 0.5,
                 paddle_gn: bool = True):
        super(SRU, self).__init__()

        # 确保 group_num 不大于 oup_channels
        if oup_channels < group_num:
            group_num = oup_channels

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if paddle_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / paddle.sum(self.gn.weight)
        w_gamma = w_gamma.reshape([1, -1, 1, 1])
        reweights = self.sigmoid(gn_x * w_gamma)
        # Gate
        w1 = paddle.where(reweights > self.gate_threshold, paddle.ones_like(reweights), reweights)  # 阈值以上置1
        w2 = paddle.where(reweights > self.gate_threshold, paddle.zeros_like(reweights), reweights)  # 阈值以上置0
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        parts = paddle.split(x_1, num_or_sections=[x_1.shape[1] // 2, x_1.shape[1] // 2], axis=1)
        x_11, x_12 = parts[0], parts[1]
        parts = paddle.split(x_2, num_or_sections=[x_2.shape[1] // 2, x_2.shape[1] // 2], axis=1)
        x_21, x_22 = parts[0], parts[1]
        return paddle.concat([x_11 + x_22, x_12 + x_21], axis=1)


class CRU(nn.Layer):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super(CRU, self).__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2D(up_channel, up_channel // squeeze_radio, kernel_size=1, bias_attr=False)
        self.squeeze2 = nn.Conv2D(low_channel, low_channel // squeeze_radio, kernel_size=1, bias_attr=False)
        # up
        self.GWC = nn.Conv2D(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2D(up_channel // squeeze_radio, op_channel, kernel_size=1, bias_attr=False)
        # low
        self.PWC2 = nn.Conv2D(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias_attr=False)
        self.advavg = nn.AdaptiveAvgPool2D(1)

    def forward(self, x):
        # Split
        up, low = x[:, :self.up_channel, :, :], x[:, self.up_channel:, :, :]
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = paddle.concat([self.PWC2(low), low], axis=1)
        # Fuse
        out = paddle.concat([Y1, Y2], axis=1)
        out = F.softmax(self.advavg(out), axis=1) * out
        # 在这里我们分割`out`，并且确保我们可以得到两个部分。
        split_sections = out.shape[1] // 2
        out_split = paddle.split(out, num_or_sections=[split_sections, split_sections], axis=1)
        out1, out2 = out_split
        return out1 + out2


class ScConv(nn.Layer):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_threshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super(ScConv, self).__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_threshold=gate_threshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        # print("应用ScConv")
        x = self.SRU(x)
        x = self.CRU(x)
        return x


if __name__ == '__main__':
    x = paddle.randn([1, 32, 16, 16])
    model = ScConv(32)
    print(model(x).shape)
