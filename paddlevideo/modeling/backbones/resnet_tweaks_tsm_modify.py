# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math

import sys
import paddle
import paddle.nn as nn
from paddle.nn import (Conv2D, BatchNorm2D, Linear, Dropout, MaxPool2D,
                       AvgPool2D)
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.regularizer import L2Decay

from ..heads.new_module import CBAM
from ..new_module_modify.C2f import C2f
from ..new_module_modify.MultiDilatelocalAttention_official import MultiDilatelocalAttention
from ..new_module_modify.ScConv_non_official import ScConv
from ..registry import BACKBONES
from ..weight_init import weight_init_
from ...utils.save_load import load_ckpt


# Download URL of pretrained model
# {
# "ResNet50_vd":
# "wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams",
# "ResNet101_vd":
# "https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ResNet101_vd_ssld_pretrained.pdparams",
# "ResNet18_vd":
# "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_vd_pretrained.pdparams",
# "ResNet34_vd":
# "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_vd_ssld_pretrained.pdparams",
# "ResNet152_vd":
# "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_vd_pretrained.pdparams",
# "ResNet200_vd":
# "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet200_vd_pretrained.pdparams",
# }


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 num_seg=8,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.num_seg = num_seg
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 act="leaky_relu",
                                 name=name + "_branch2a")
        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 act=None,
                                 name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                     name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        # add temporal shift module
        shifts = F.temporal_shift(inputs, self.num_seg, 1.0 / self.num_seg)
        y = self.conv0(shifts)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(short, conv1)
        y = F.leaky_relu(y)
        return y


class ConvBNLayer(nn.Layer):
    """Conv2D and BatchNorm2D layer.
    Args:
        in_channels (int): Number of channels for the input.
        out_channels (int): Number of channels for the output.
        kernel_size (int): Kernel size.
        stride (int): Stride in the Conv2D layer. Default: 1.
        groups (int): Groups in the Conv2D, Default: 1.
        is_tweaks_mode (bool): switch for tweaks. Default: False.
        act (str): Indicate activation after BatchNorm2D layer.
        name (str): the name of an instance of ConvBNLayer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 is_tweaks_mode=False,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.is_tweaks_mode = is_tweaks_mode
        # ResNet-D 1/2:add a 2×2 average pooling layer with a stride of 2 before the convolution,
        #             whose stride is changed to 1, works well in practice.
        self._pool2d_avg = AvgPool2D(kernel_size=2,
                                     stride=2,
                                     padding=0,
                                     ceil_mode=True)

        self._conv = Conv2D(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2,
                            groups=groups,
                            weight_attr=ParamAttr(name=name + "_weights"),
                            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        self._act = act

        self._batch_norm = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(name=bn_name + "_scale",
                                  regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(bn_name + "_offset", regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        if self.is_tweaks_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._act:
            y = getattr(paddle.nn.functional, self._act)(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 num_seg=8,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 act="leaky_relu",
                                 name=name + "_branch2a")
        # self.conv1 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels,
        #                          kernel_size=3,
        #                          stride=stride,
        #                          act="leaky_relu",
        #                          name=name + "_branch2b")
        # self.conv2 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels * 4,
        #                          kernel_size=1,
        #                          act=None,
        #                          name=name + "_branch2c")

        # todo: 应用C2f
        self.conv1 = C2f(c1=out_channels, c2=out_channels)
        self.conv2 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels * 4,
                                 kernel_size=1,
                                 stride=stride,
                                 act=None,
                                 name=name + "_branch2c")

        # # todo:将每个阶段中每个块内的3x3卷积换成1个scconv   +  head中使用 MultiDilatelocalAttention         (top1 acc)0.85
        # self.conv1 = ScConv(op_channel=out_channels, group_num=4)
        # self.conv2 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels * 4,
        #                          kernel_size=1,
        #                          stride=stride,
        #                          act=None,
        #                          name=name + "_branch2c")

        # # todo:将每个阶段中每个块内的3x3卷积换成1个scconv    (top1 acc)0.85
        # self.conv1 = ScConv(op_channel=out_channels, group_num=4)
        # self.conv2 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels * 4,
        #                          kernel_size=1,
        #                          stride=stride,
        #                          act=None,
        #                          name=name + "_branch2c")

        # # todo:将每个阶段中每个块内的3x3卷积换成1个scconv  + CBMA模块     (top1 acc)0.85
        # self.conv1 = ScConv(op_channel=out_channels, group_num=4)
        # self.cbam = CBAM(out_channels)
        # self.conv2 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels * 4,
        #                          kernel_size=1,
        #                          stride=stride,
        #                          act=None,
        #                          name=name + "_branch2c")

        # # todo:CBMA模块 + 将每个阶段中每个块内的3x3卷积换成1个scconv      (top1 acc)0.85
        # self.cbam = CBAM(out_channels)
        # self.conv1 = ScConv(op_channel=out_channels, group_num=4)
        # self.conv2 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels * 4,
        #                          kernel_size=1,
        #                          stride=stride,
        #                          act=None,
        #                          name=name + "_branch2c")

        # # todo:每个阶段中每个块内的3x3卷积，换成2个scconv   (top1 acc)0.75
        # self.conv1 = ScConv(op_channel=out_channels, group_num=4)
        # self.conv2 = ScConv(op_channel=out_channels, group_num=4)
        # self.conv3 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels * 4,
        #                          kernel_size=1,
        #                          stride=stride,
        #                          act=None,
        #                          name=name + "_branch2c")

        # # todo:将每个阶段中每个块内的3x3卷积换成1个 MultiDilatelocalAttention
        # # x = paddle.randn([1, 32, 16, 16])
        # # model = MultiDilatelocalAttention(dim=16, num_heads=8, dilation=[1, 2])
        # print("out_channels:", out_channels)

        # self.conv2 = ConvBNLayer(in_channels=out_channels,
        #                          out_channels=out_channels * 4,
        #                          kernel_size=1,
        #                          act=None,
        #                          name=name + "_branch2c")

        # self.conv2 = MultiDilatelocalAttention(dim=80, num_heads=8, dilation=[1, 2])


        # # todo :新加模块 【ScConv】
        # self.scconv = ScConv(out_channels * 4)  # 添加 ScConv 模块  # Bottleneck 架构中的每个残差块（Residual Block）的最后一层卷积输出的通道数为输入通道数的 4 倍。

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels * 4,
                                     kernel_size=1,
                                     stride=1,
                                     is_tweaks_mode=False if if_first else True,
                                     name=name + "_branch1")

        self.shortcut = shortcut
        self.num_seg = num_seg



    def forward(self, inputs):
        if paddle.is_compiled_with_custom_device('npu'):
            x = inputs
            seg_num = self.num_seg
            shift_ratio = 1.0 / self.num_seg

            shape = x.shape  # [N*T, C, H, W]
            reshape_x = x.reshape(
                (-1, seg_num, shape[1], shape[2], shape[3]))  # [N, T, C, H, W]
            pad_x = F.pad(reshape_x, [
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ])  # [N, T+2, C, H, W]
            c1 = int(shape[1] * shift_ratio)
            c2 = int(shape[1] * 2 * shift_ratio)
            slice1 = pad_x[:, :seg_num, :c1, :, :]
            slice2 = pad_x[:, 2:seg_num + 2, c1:c2, :, :]
            slice3 = pad_x[:, 1:seg_num + 1, c2:, :, :]
            concat_x = paddle.concat([slice1, slice2, slice3],
                                     axis=2)  # [N, T, C, H, W]
            shifts = concat_x.reshape(shape)
        else:
            shifts = F.temporal_shift(inputs, self.num_seg, 1.0 / self.num_seg)

        y = self.conv0(shifts)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # # # todo:将每个阶段中每个块内的3x3卷积换成1个 MultiDilatelocalAttention
        # y = self.conv0(shifts)
        # print("y.shape:", y.shape)
        # conv1 = self.conv1(y)
        # print("conv1:", conv1.shape)
        # conv2 = self.conv2(conv1)
        # print("conv2:", conv2.shape)
        # # conv2 = self.conv3(conv2)

        # #  todo :应用scconv空间和通道重建卷积  (应用scconv模块到shifts操作后的输出)    # bt=1  (top1 acc)0.6
        # shifts_after_scconv = self.scconv(shifts)
        # y = self.conv0(shifts_after_scconv)

        # # todo:将每个阶段中每个块内的3x3卷积换成1个scconv
        # y = self.conv0(shifts)
        # conv1 = self.conv1(y)
        # conv2 = self.conv2(conv1)

        # # todo:将每个阶段中每个块内的3x3卷积换成1个scconv  + CBMA模块
        # y = self.conv0(shifts)
        # conv1 = self.conv1(y)
        # conv1 = self.cbam(conv1)
        # conv2 = self.conv2(conv1)

        # # todo: CBMA模块 + 将每个阶段中每个块内的3x3卷积换成1个scconv
        # y = self.conv0(shifts)
        # conv1 = self.cbam(y)
        # conv1 = self.conv1(conv1)
        # conv2 = self.conv2(conv1)

        # # todo:每个阶段中每个块内的3x3卷积，换成2个scconv
        # y = self.conv0(shifts)
        # conv1 = self.conv1(y)
        # conv2 = self.conv2(conv1)
        # conv2 = self.conv3(conv2)

        #  todo :应用scconv空间和通道重建卷积      # bt=1  太差
        # conv2 = self.scconv(conv2)


        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        return F.leaky_relu(y)


@BACKBONES.register()
class ResNetTweaksTSMModify(nn.Layer):
    """ResNet TSM backbone.

    Args:
        depth (int): Depth of resnet model.
        pretrained (str): pretrained model. Default: None.
    """

    def __init__(self, depth, num_seg=8, pretrained=None):
        super(ResNetTweaksTSMModify, self).__init__()
        self.pretrained = pretrained
        self.layers = depth
        self.num_seg = num_seg

        supported_layers = [18, 34, 50, 101, 152]
        assert self.layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, self.layers)

        if self.layers == 18:
            depth = [2, 2, 2, 2]
        elif self.layers == 34 or self.layers == 50:
            depth = [3, 4, 6, 3]
        elif self.layers == 101:
            depth = [3, 4, 23, 3]
        elif self.layers == 152:
            depth = [3, 8, 36, 3]

        in_channels = 64
        out_channels = [64, 128, 256, 512]

        # # ResNet-C: use three 3x3 conv, replace, one 7x7 conv
        self.conv1_1 = ConvBNLayer(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=2,
                                   act='leaky_relu',
                                   name="conv1_1")
        # self.conv1_2 = ConvBNLayer(in_channels=32,
        #                            out_channels=32,
        #                            kernel_size=3,
        #                            stride=1,
        #                            act='leaky_relu',
        #                            name="conv1_2")
        # self.conv1_3 = ConvBNLayer(in_channels=32,
        #                            out_channels=64,
        #                            kernel_size=3,
        #                            stride=1,
        #                            act='leaky_relu',
        #                            name="conv1_3")

        # todo: 第二个3x3卷积，换C2f [ 将每个阶段中每个块内的3x3卷积换成1个C2f + input stem中，第二个3x3卷积，换C2f  ==> (top1 acc)0.9 ]
        self.conv1_2 = C2f(c1=32, c2=32)
        self.conv1_3 = ConvBNLayer(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   act='leaky_relu',
                                   name="conv1_3")

        # # todo:在input stem中，用一个scconv换一个3x3卷积  + CBAM
        # # 假设每个ScConv模块的输出通道数与原来的ConvBNLayer一致
        # self.conv1_2 = ScConv(op_channel=32, group_num=4)  # Adjust the parameters as necessary
        # self.cbam = CBAM(32)
        # self.conv1_3 = ConvBNLayer(in_channels=32,
        #                            out_channels=64,
        #                            kernel_size=3,
        #                            stride=1,
        #                            act='leaky_relu',
        #                            name="conv1_3")

        # # todo:在input stem中，用一个scconv换一个3x3卷积      (top1 acc)0.8
        # # 假设每个ScConv模块的输出通道数与原来的ConvBNLayer一致
        # self.conv1_2 = ScConv(op_channel=32, group_num=4)  # Adjust the parameters as necessary
        # self.conv1_3 = ConvBNLayer(in_channels=32,
        #                            out_channels=64,
        #                            kernel_size=3,
        #                            stride=1,
        #                            act='leaky_relu',
        #                            name="conv1_3")

        # # todo:在input stem中，用2个scconv换一个3x3卷积    (top1 acc)0.65
        # self.conv1_2 = ScConv(op_channel=32, group_num=4)
        # self.conv1_3 = ScConv(op_channel=32, group_num=4)
        # self.conv1_4 = ConvBNLayer(in_channels=32,
        #                            out_channels=64,
        #                            kernel_size=3,
        #                            stride=1,
        #                            act='leaky_relu',
        #                            name="conv1_3")



        self.pool2D_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if self.layers >= 50:
            for block in range(len(depth)):  # 对于模型中的每个块，迭代创建多个层。depth是一个列表，其中每个元素指定相应块中层的数量。 depth = [3, 4, 6, 3]
                shortcut = False
                for i in range(depth[block]):  # 对于每个块内的每个层，迭代构建层。
                    if self.layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)  # 每个卷积层的命名方式，这里使用block编号和层序号来构造名称
                    bottleneck_block = self.add_sublayer(  # 创建一个瓶颈块（BottleneckBlock）并将其添加到模型中作为一个子层。
                        'bb_%d_%d' %
                        (block, i),  # same with PaddleClas, for loading pretrain
                        BottleneckBlock(
                            in_channels=in_channels
                            if i == 0 else out_channels[block] * 4,
                            out_channels=out_channels[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            num_seg=self.num_seg,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    in_channels = out_channels[block] * 4
                    self.block_list.append(bottleneck_block)  # 将新创建的瓶颈块添加到之前初始化的块列表中
                    shortcut = True  # 在添加了第一个层后，设置shortcut为True，这样后续的层都会有shortcut连接。
        else:
            in_channels = [64, 64, 128, 256]
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(in_channels=in_channels[block]
                        if i == 0 else out_channels[block],
                                   out_channels=out_channels[block],
                                   stride=2 if i == 0 and block != 0 else 1,
                                   shortcut=shortcut,
                                   num_seg=self.num_seg,
                                   name=conv_name))
                    self.block_list.append(basic_block)
                    shortcut = True

        # # 在__init__方法的末尾添加以下代码
        # print("Total blocks in block_list:", len(self.block_list))
        # for idx, blk in enumerate(self.block_list):
        #     print(f"Block index {idx}: {blk}")

    def init_weights(self):
        """Initiate the parameters.
        Note:
            1. when indicate pretrained loading path, will load it to initiate backbone.
            2. when not indicating pretrained loading path, will follow specific initialization initiate backbone. Always, Conv2D layer will be initiated by KaimingNormal function, and BatchNorm2d will be initiated by Constant function.
            Please refer to https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/initializer/kaiming/KaimingNormal_en.html
        """
        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    # no bias
                    weight_init_(layer, 'KaimingNormal')
                elif isinstance(layer, nn.BatchNorm2D):
                    weight_init_(layer, 'Constant', value=1)

    def forward(self, inputs):
        """Define how the backbone is going to run.
        """
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)

        # # todo: 一个scconv换一个3x3卷积 + scconv后加一个CBAM
        # y = self.conv1_1(inputs)
        # y = self.conv1_2(y)
        # y = self.cbam(y)
        # y = self.conv1_3(y)

        # # todo:在input stem中，用2个scconv换一个3x3卷积
        # y = self.conv1_4(y)

        y = self.pool2D_max(y)
        for block in self.block_list:
            y = block(y)
        return y


    # def forward(self, inputs):
    #     # 在每个阶段之前应用卷积层，以匹配维度
    #     def match_dimensions(x, y):
    #         # 检查通道数是否相同
    #         if x.shape[1] != y.shape[1]:
    #             # 调整通道数
    #             x = nn.Conv2D(in_channels=x.shape[1], out_channels=y.shape[1], kernel_size=1)(x)
    #         # 检查空间维度是否相同
    #         if x.shape[2] != y.shape[2] or x.shape[3] != y.shape[3]:
    #             # 应用平均池化以减少空间维度
    #             x = nn.AvgPool2D(kernel_size=2, stride=2)(x)
    #         return x
    #
    #     y = self.conv1_1(inputs)
    #     y = self.conv1_2(y)
    #     y = self.conv1_3(y)
    #     y = self.pool2D_max(y)
    #
    #     # 初始化变量，用于保存每个阶段开始的特征
    #     stage_feature = y
    #
    #     # 将block_list重新组织成一个嵌套列表，每个子列表包含一个阶段的所有块
    #     stage_blocks = [self.block_list[:3], self.block_list[3:7], self.block_list[7:13], self.block_list[13:]]
    #
    #     for idx, blocks in enumerate(stage_blocks):
    #         # 对于每个阶段内的每个块
    #         for block in blocks:
    #             y = block(y)
    #         # 调整阶段开始特征的维度以匹配阶段结束特征的维度
    #         stage_feature = match_dimensions(stage_feature, y)
    #         # 将调整维度后的阶段开始特征加到当前特征图上
    #         y = stage_feature + y
    #         if idx < len(stage_blocks) - 1:
    #             # 更新阶段特征，用于下一阶段
    #             stage_feature = y
    #
    #     return y

