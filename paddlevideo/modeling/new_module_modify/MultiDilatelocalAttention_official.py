# https://github.com/JIAOJIAYUASD/dilateformer/blob/main/models/dilateformer.py

# 引入 PaddlePaddle
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.modeling import DropPath, to_2tuple


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 注意：这里假设MultiDilatelocalAttention和Mlp已经被正确地转换为PaddlePaddle的类。
# 如果没有，需要对应转换这些类。
class DilateBlock(nn.Layer):
    "Implementation of Dilate-attention block in Paddle"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.transpose([0, 2, 3, 1])
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose([0, 3, 1, 2])
        return x


class Identity(nn.Layer):
    def forward(self, x):
        return x


class GlobalAttention(nn.Layer):
    """Implementation of self-attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape([B, H * W, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = paddle.matmul(q, k, transpose_y=True) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v).transpose([0, 2, 1, 3])
        x = x.reshape([B, H, W, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 假设GlobalAttention和Mlp已正确转换为PaddlePaddle类，且DropPath是自定义的或者找到的等价替代PaddlePaddle类
class GlobalBlock(nn.Layer):
    """
    Implementation of Transformer
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop)

        # DropPath 在Paddle中没有直接的等价物，可能需要自定义或者寻找替代方案
        # 而 nn.Identity 在Paddle中可以用 nn.Layer 实现，但通常用在组合Layer中不需要的时候
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Layer()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = paddle.transpose(x, [0, 2, 3, 1])
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = paddle.transpose(x, [0, 3, 1, 2])
        return x


# Note: Assuming to_2tuple is a function that converts a value or a tuple into a tuple of two elements.
# In PaddlePaddle, such utility function might not exist and should be implemented if needed.

class PatchEmbed(nn.Layer):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96, patch_way=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 导入或定义相应的to_2tuple函数
        patch_size = to_2tuple(patch_size)  # 导入或定义相应的to_2tuple函数
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'], \
            "the patch embedding way isn't exist!"

        if patch_way == "nonoverlaping":
            self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                nn.Conv2D(in_chans, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.GELU(),
                nn.Conv2D(hidden_dim, int(hidden_dim * 2), kernel_size=3, stride=2,
                          padding=1, bias_attr=False),
                nn.BatchNorm2D(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2D(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=1,
                          padding=1, bias_attr=False),
                nn.BatchNorm2D(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2D(int(hidden_dim * 4), embed_dim, kernel_size=3, stride=2,
                          padding=1, bias_attr=False),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2D(in_chans, hidden_dim, kernel_size=3, stride=2,
                          padding=1, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.GELU(),
                nn.Conv2D(hidden_dim, int(hidden_dim * 2), kernel_size=1, stride=1,
                          padding=0, bias_attr=False),
                nn.BatchNorm2D(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2D(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=2,
                          padding=1, bias_attr=False),
                nn.BatchNorm2D(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2D(int(hidden_dim * 4), embed_dim, kernel_size=1, stride=1,
                          padding=0, bias_attr=False),
            )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class PatchMerging(nn.Layer):
    """Patch Merging Layer."""

    def __init__(self, in_channels, out_channels, merging_way, cpe_per_stage, norm_layer=nn.BatchNorm2D):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_stage = cpe_per_stage

        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                # 注意：在Paddle中AvgPool2d的第一个参数是kernel_size，而不是输入通道数。
                nn.AvgPool2D(kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        else:  # avgpool2_2
            self.proj = nn.Sequential(
                # 同样，这里的AvgPool2D的第一个参数是kernel_size。
                nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )

        if self.cpe_per_stage:
            self.pos_embed = nn.Conv2D(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        # x: B, C, H, W
        x = self.proj(x)
        if self.cpe_per_stage:
            x = x + self.pos_embed(x)
        return x


# 注意: 如果需要，还应该定义或导入to_2tuple函数。


# 假设DilateBlock和PatchMerging已经被转换成PaddlePaddle类
class DilateStage(nn.Layer):
    """A basic Dilate Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_stage=False, cpe_per_block=False,
                 downsample=True, merging_way=None):

        super().__init__()
        # build blocks
        self.blocks = nn.LayerList([
            DilateBlock(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_stage)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


# 注意: 这里的代码假设DilateBlock和PatchMerging类已经被转换为PaddlePaddle兼容的版本。
# nn.Identity在Paddle中直接使用nn.Layer()来替换。


# 假设GlobalBlock和PatchMerging已经被转换成PaddlePaddle类
class GlobalStage(nn.Layer):
    """A basic Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_stage=False, cpe_per_block=False,
                 downsample=True, merging_way=None):

        super().__init__()
        # build blocks
        self.blocks = nn.LayerList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_stage)
        else:
            self.downsample = nn.Identity()  # In PaddlePaddle, use nn.Layer() as the identity layer

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


# 注意: 这里的代码假设GlobalBlock和PatchMerging类已经被转换为PaddlePaddle兼容的版本。
# 在PaddlePaddle中，nn.Identity()可以使用nn.Layer()实现。


# 假设PatchEmbed, DilateStage, GlobalStage都已经被转换成PaddlePaddle类
class Dilateformer(nn.Layer):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], kernel_size=3, dilation=[1, 2, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 dilate_attention=[True, True, False, False],
                 downsamples=[True, True, True, False],
                 cpe_per_stage=False, cpe_per_block=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)

        # patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)
        dpr = [x for x in paddle.linspace(0, drop_path, sum(depths))]
        self.stages = nn.LayerList()
        for i_layer in range(self.num_layers):
            if dilate_attention[i_layer]:
                stage = DilateStage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_stage=cpe_per_stage,
                                    merging_way=merging_way
                                    )
            else:
                stage = GlobalStage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_stage=cpe_per_stage,
                                    merging_way=merging_way
                                    )
            self.stages.append(stage)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32',
                                               default_initializer=nn.initializer.XavierUniform())
            m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32',
                                             default_initializer=nn.initializer.Constant(value=0.0))
        elif isinstance(m, nn.LayerNorm):
            m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32',
                                             default_initializer=nn.initializer.Constant(value=0.0))
            m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32',
                                               default_initializer=nn.initializer.Constant(value=1.0))

    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)

        x = x.flatten(2).transpose((0, 2, 1))
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose((0, 2, 1)))  # B C 1
        x = paddle.flatten(x, start_axis=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# # 假设已经定义了Dilateformer类和_cfg函数


def dilateformer_small(pretrained=True, **kwargs):
    model = Dilateformer(depths=[3, 5, 8, 3], embed_dim=72, num_heads=[3, 6, 12, 24], **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        # 加载预训练模型权重
        pass  # 未实现：在此处添加适用于PaddlePaddle的权重加载代码
    return model


def dilateformer_base(pretrained=True, **kwargs):
    model = Dilateformer(depths=[4, 8, 10, 3], embed_dim=96, num_heads=[3, 6, 12, 24], **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        # 加载预训练模型权重
        pass  # 未实现：在此处添加适用于PaddlePaddle的权重加载代码
    return model


def dilateformer_tiny(pretrained=True, **kwargs):
    model = Dilateformer(depths=[2, 2, 6, 2], embed_dim=72, num_heads=[3, 6, 12, 24], **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        # 加载预训练模型权重
        pass  # 未实现：在此处添加适用于PaddlePaddle的权重加载代码
    return model

# todo： 原代码   (top1 acc)0.95
class DilateAttention(nn.Layer):
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        q = q.reshape([B, int(d // self.head_dim), int(self.head_dim), 1, H * W]).transpose([0, 1, 4, 3, 2])  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).transpose([0, 1, 4, 2, 3])  # B,h,N,d,k*k
        attn = paddle.matmul(q, k) * self.scale  # B,h,N,1,k*k
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).transpose([0, 1, 4, 3, 2])  # B,h,N,k*k,d
        x = paddle.matmul(attn, v).transpose([0, 2, 3, 1, 4]).reshape([B, H, W, d])
        return x


class MultiDilatelocalAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2D(dim, dim * 3, 1, bias_attr=qkv_bias)
        self.dilate_attention = nn.LayerList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
            for i in range(self.num_dilation)
        ])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B, H, W, C = x.shape
        B, C, H, W = x.shape
        # print("B, H, W, C:", B, H, W, C)
        # print("x.shape:", x.shape)      # [1, 32, 16, 16]
        # x = x.transpose([0, 2, 1, 3])  # B, C, H, W
        qkv = self.qkv(x).reshape([B, 3, self.num_dilation, C // self.num_dilation, H, W]).transpose([2, 1, 0, 3, 4, 5])
        # num_dilation, 3, B, C//num_dilation, H, W
        x = x.reshape([B, self.num_dilation, C // self.num_dilation, H, W]).transpose([1, 0, 3, 4, 2])
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W, C//num_dilation
        x = x.transpose([1, 2, 3, 0, 4]).reshape([B, H, W, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

 # todo： 修改qkv层以使用组卷积        (top1 acc)0.9
# class MultiDilatelocalAttention(nn.Layer):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3], groups=1):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.groups = groups  # 添加分组参数
#         head_dim = dim // num_heads
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_dilation = len(dilation)
#         assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
#         # self.qkv = nn.Conv2D(dim, dim * 3, 1, bias_attr=qkv_bias)
#         # todo： 修改qkv层以使用组卷积        (top1 acc)0.9
#         self.qkv = nn.Conv2D(dim, dim * 3, 1, groups=self.groups, bias_attr=qkv_bias)
#         self.dilate_attention = nn.LayerList([
#             DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
#             for i in range(self.num_dilation)
#         ])
#         # self.proj = nn.Linear(dim, dim)
#         # 修改投影层以适应组卷积
#         self.proj = nn.Linear(dim * self.groups, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         # B, H, W, C = x.shape
#         B, C, H, W = x.shape
#         # print("B, H, W, C:", B, H, W, C)
#         # print("x.shape:", x.shape)      # [1, 32, 16, 16]
#         # x = x.transpose([0, 2, 1, 3])  # B, C, H, W
#         qkv = self.qkv(x).reshape([B, 3, self.num_dilation, C // self.num_dilation, H, W]).transpose([2, 1, 0, 3, 4, 5])
#         # num_dilation, 3, B, C//num_dilation, H, W
#         x = x.reshape([B, self.num_dilation, C // self.num_dilation, H, W]).transpose([1, 0, 3, 4, 2])
#         # num_dilation, B, H, W, C//num_dilation
#         for i in range(self.num_dilation):
#             x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W, C//num_dilation
#         x = x.transpose([1, 2, 3, 0, 4]).reshape([B, H, W, C])
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# # todo:在注意力计算之前应用层归一化（Layer Normalization）===> 提升性能
# class DilateAttention(nn.Layer):
#     def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
#         super().__init__()
#         self.head_dim = head_dim
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kernel_size = kernel_size
#         self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def forward(self, q, k, v):
#         B, d, H, W = q.shape
#         q = q.reshape([B, int(d // self.head_dim), int(self.head_dim), 1, H * W]).transpose([0, 1, 4, 3, 2])  # B,h,N,1,d
#         k = self.unfold(k).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).transpose([0, 1, 4, 2, 3])  # B,h,N,d,k*k
#         attn = paddle.matmul(q, k) * self.scale  # B,h,N,1,k*k
#         attn = paddle.nn.functional.softmax(attn, axis=-1)
#         attn = self.attn_drop(attn)
#         v = self.unfold(v).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).transpose([0, 1, 4, 3, 2])  # B,h,N,k*k,d
#         x = paddle.matmul(attn, v).transpose([0, 2, 3, 1, 4]).reshape([B, H, W, d])
#         return x
#
# class MultiDilatelocalAttention(nn.Layer):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_dilation = len(dilation)
#         assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
#         self.qkv = nn.Conv2D(dim, dim * 3, 1, bias_attr=qkv_bias)
#         self.dilate_attention = nn.LayerList([
#             DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
#             for i in range(self.num_dilation)
#         ])
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         # todo：添加归一化
#         x = paddle.nn.functional.normalize(x, p=2, axis=1)
#
#         # B, H, W, C = x.shape
#         B, C, H, W = x.shape
#         # print("B, H, W, C:", B, H, W, C)
#         # print("x.shape:", x.shape)      # [1, 32, 16, 16]
#         # x = x.transpose([0, 2, 1, 3])  # B, C, H, W
#         qkv = self.qkv(x).reshape([B, 3, self.num_dilation, C // self.num_dilation, H, W]).transpose([2, 1, 0, 3, 4, 5])
#         # num_dilation, 3, B, C//num_dilation, H, W
#         x = x.reshape([B, self.num_dilation, C // self.num_dilation, H, W]).transpose([1, 0, 3, 4, 2])
#         # num_dilation, B, H, W, C//num_dilation
#         for i in range(self.num_dilation):
#             x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W, C//num_dilation
#         x = x.transpose([1, 2, 3, 0, 4]).reshape([B, H, W, C])
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# # todo: 在DilateAttention前后添加卷积层
# class DilateAttention(nn.Layer):
#     def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
#         super().__init__()
#         self.head_dim = head_dim
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kernel_size = kernel_size
#         self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
#         self.attn_drop = nn.Dropout(attn_drop)
#
#         # 定义新的卷积层
#         # # todo: 添加1x1卷积层    (top1 acc)0.9
#         # # print("head_dim:", head_dim)
#         # # print("head_dim * 4:", head_dim * 4)
#         # self.conv_q = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=1, padding=0)
#         # self.conv_k = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=1, padding=0)
#         # self.conv_v = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=1, padding=0)
#         # self.conv_out = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=1, padding=0)
#
#         # todo: 添加3x3卷积层    [ loss:     nan ]
#         # print("head_dim:", head_dim)
#         # print("head_dim * 4:", head_dim * 4)
#         self.conv_q = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=3, padding=1, stride=1)
#         self.conv_k = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=3, padding=1, stride=1)
#         self.conv_v = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=3, padding=1, stride=1)
#         self.conv_out = nn.Conv2D(head_dim * 4, head_dim * 4, kernel_size=3, padding=1, stride=1)
#
#     def forward(self, q, k, v):
#         # print("q.shape:", q.shape)
#         # print("k.shape:", k.shape)
#         # print("v.shape:", v.shape)
#         # 在这里添加输入的卷积层
#         q = self.conv_q(q)  # 假设conv_q是在__init__中定义的卷积层
#         k = self.conv_k(k)  # 假设conv_k是在__init__中定义的卷积层
#         v = self.conv_v(v)  # 假设conv_v是在__init__中定义的卷积层
#
#         B, d, H, W = q.shape
#         q = q.reshape([B, int(d // self.head_dim), int(self.head_dim), 1, H * W]).transpose([0, 1, 4, 3, 2])  # B,h,N,1,d
#         k = self.unfold(k).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).transpose([0, 1, 4, 2, 3])  # B,h,N,d,k*k
#         attn = paddle.matmul(q, k) * self.scale  # B,h,N,1,k*k
#         attn = paddle.nn.functional.softmax(attn, axis=-1)
#         attn = self.attn_drop(attn)
#         v = self.unfold(v).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).transpose([0, 1, 4, 3, 2])  # B,h,N,k*k,d
#         x = paddle.matmul(attn, v).transpose([0, 2, 3, 1, 4]).reshape([B, H, W, d])
#
#         # 在这里添加输出的卷积层
#         # print("x.shape:", x.shape)
#         # print("x.reshape:", x.transpose([0, 3, 1, 2]).shape)
#         x = self.conv_out(x.transpose([0, 3, 1, 2]))  # 假设conv_out是在__init__中定义的卷积层
#         x = x.transpose([0, 2, 3, 1])
#         # print("x.shape2:", x.shape)
#         return x
#
#
# class MultiDilatelocalAttention(nn.Layer):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_dilation = len(dilation)
#         assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
#         self.qkv = nn.Conv2D(dim, dim * 3, 1, bias_attr=qkv_bias)
#         self.dilate_attention = nn.LayerList([
#             DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
#             for i in range(self.num_dilation)
#         ])
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         # B, H, W, C = x.shape
#         B, C, H, W = x.shape
#         # print("B, H, W, C:", B, H, W, C)
#         # print("x.shape:", x.shape)      # [1, 32, 16, 16]
#         # x = x.transpose([0, 2, 1, 3])  # B, C, H, W
#         qkv = self.qkv(x).reshape([B, 3, self.num_dilation, C // self.num_dilation, H, W]).transpose([2, 1, 0, 3, 4, 5])
#         # num_dilation, 3, B, C//num_dilation, H, W
#         x = x.reshape([B, self.num_dilation, C // self.num_dilation, H, W]).transpose([1, 0, 3, 4, 2])
#         # num_dilation, B, H, W, C//num_dilation
#         for i in range(self.num_dilation):
#             x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W, C//num_dilation
#         x = x.transpose([1, 2, 3, 0, 4]).reshape([B, H, W, C])
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x



if __name__ == "__main__":
    x = paddle.randn([1, 32, 16, 16])
    model = MultiDilatelocalAttention(dim=32, num_heads=8, dilation=[1, 2])
    # print("model(x).shape:", model(x).shape)      # [1, 16, 16, 32]
    print("model(x).transpose([0,3,1,2]).shape:", model(x).transpose([0,3,1,2]).shape)  # [1, 32, 16, 16]
