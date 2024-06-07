# https://blog.csdn.net/m0_63774211/article/details/129493630

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# # 原来
# class v8_C2fBottleneck_orgin(nn.Layer):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels     # e: 扩展因子，用于计算隐藏层通道数。
#
#         padding_1 = k[0] // 2 if isinstance(k[0], int) else (k[0][0] // 2, k[0][1] // 2)
#         padding_2 = k[1] // 2 if isinstance(k[1], int) else (k[1][0] // 2, k[1][1] // 2)
#
#         self.cv1 = nn.Conv2D(c1, c_, k[0], 1, padding=padding_1)
#         self.cv2 = nn.Conv2D(c_, c2, k[1], 1, groups=g, padding=padding_2)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
#
# class C2f(nn.Layer):
#     # CSP Bottleneck with 2 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = nn.Conv2D(c1, 2 * self.c, 1, 1)
#         self.cv2 = nn.Conv2D((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.LayerList(
#             v8_C2fBottleneck_orgin(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(paddle.concat(y, axis=1))

# todo：修改
# todo：修改一：(top1 acc)0.9		【 epoch: 15 】
# class v8_C2fBottleneck(nn.Layer):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, dropout_rate=0.1):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels     # e: 扩展因子，用于计算隐藏层通道数。
#
#         padding_1 = k[0] // 2 if isinstance(k[0], int) else (k[0][0] // 2, k[0][1] // 2)
#         padding_2 = k[1] // 2 if isinstance(k[1], int) else (k[1][0] // 2, k[1][1] // 2)
#
#         # First Convolution + Batch Normalization + ReLU Activation + Dropout
#         self.cv1 = nn.Sequential(
#             nn.Conv2D(c1, c_, k[0], 1, padding=padding_1),
#             nn.BatchNorm2D(c_),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
#         # Second Convolution + Batch Normalization + ReLU Activation + Dropout
#         self.cv2 = nn.Sequential(
#             nn.Conv2D(c_, c2, k[1], 1, groups=g, padding=padding_2),
#             nn.BatchNorm2D(c2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         out = self.cv1(x)
#         out = self.cv2(out)
#         return x + out if self.add else out

# todo：修改二：
class v8_C2fBottleneck(nn.Layer):
    # Enhanced bottleneck with Depthwise Separable Convolution, Batch Normalization, and ReLU activation
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        padding_1 = k[0] // 2 if isinstance(k[0], int) else (k[0][0] // 2, k[0][1] // 2)
        padding_2 = k[1] // 2 if isinstance(k[1], int) else (k[1][0] // 2, k[1][1] // 2)

        # Depthwise Convolution
        self.dwconv = nn.Sequential(
            nn.Conv2D(c1, c1, kernel_size=k[0], stride=1, padding=padding_1, groups=c1),
            nn.BatchNorm2D(c1),
            nn.ReLU()
        )

        # Pointwise Convolution
        self.pwconv1 = nn.Sequential(
            nn.Conv2D(c1, c_, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(c_),
            nn.ReLU()
        )
        self.pwconv2 = nn.Sequential(
            nn.Conv2D(c_, c2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(c2),
            nn.ReLU()
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        return residual + x if self.add else x


class C2f(nn.Layer):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2D(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2D((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.LayerList(
            v8_C2fBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(paddle.concat(y, axis=1))



