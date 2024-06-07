import paddle
from paddle.nn import Linear

# 假设我们有一个简单的模型
model = paddle.nn.Sequential(
    Linear(10, 20),  # 第一个全连接层
    Linear(20, 10)   # 第二个全连接层
)

# 计算参数量
num_params = sum(p.numel() for p in model.parameters() if p.stop_gradient is False)
print(f'The model has {num_params} parameters.')  # 打印参数量