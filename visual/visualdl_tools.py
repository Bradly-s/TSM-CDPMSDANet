# import paddle
# from visualdl import LogWriter
#
# # 创建一个LogWriter对象

#
# log_writer = LogWriter(logdir="./visualdl_log/ResNetTweaksTSM", sync_cycle=100)
#
# # 选择要可视化的模型
# model = ResNetTweaksTSM(depth=50) # 例如，使用ResNet50
# model.eval() # 确保模型处于评估模式，这样不会记录梯度等
#
# # 获取一个随机的输入Tensor
# fake_input = paddle.randn([1, 3, 340, 340]) # 假设输入是1个RGB图像，大小为224x224
# fake_input = [fake_input]  # 将其放入列表中
#
# # 通过model和fake_input来构建计算图，并记录下来
# log_writer.add_graph(model, fake_input)

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from visualdl import LogWriter
from paddlevideo.modeling.backbones import ResNetTweaksTSM

net = ResNetTweaksTSM(depth=50)
with LogWriter(logdir="./log/graph_test/") as writer:
    writer.add_graph(
        model=net,
        input_spec=[paddle.static.InputSpec([1, 3, 340, 340], 'float32')],
        verbose=True)
