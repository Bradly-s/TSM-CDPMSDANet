# https://aistudio.baidu.com/projectdetail/2806195

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定只使用第一张显卡进行训练
from PIL import Image
import paddle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import paddle.nn.functional as F
import paddle
import warnings

from paddlevideo.modeling.backbones import ResNetTweaksTSMModify

warnings.filterwarnings('ignore')

# 定义一个函数来注册梯度钩子并保存梯度
global features_grad
def extract_features_grad(grad):
    global features_grad
    features_grad = grad

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)  # Image.BILINEAR双线性插值
    if transform:
        img = transform(img)
    # img = img.unsqueeze(0)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = paddle.to_tensor(img)
    img = paddle.unsqueeze(img, axis=0)
    # print(img.shape)
    # 获取模型输出的feature/score

    # 创建一个批次大小为8的批次
    batch_imgs = paddle.concat([img] * 8, axis=0)  # 假设你希望批次大小为8
    # 使用修改后的批次大小
    # output, features = model(batch_imgs)
    output, features, *_ = model(batch_imgs)  # 如果返回了两个或更多的值

    # 打印输出和特征图的形状
    print('outputshape:', output.shape)
    print('featureshape:', features.shape)

    # 注册特征图的梯度钩子
    features.register_hook(extract_features_grad)

    # 获取预测得分最高的类别对应的输出score
    pred_class_idx = np.argmax(output.numpy(), axis=1)[0]
    pred_class = output[0, pred_class_idx]

    # 通过类别得分执行反向传播，计算梯度
    pred_class.backward()

    # 获取特征图的梯度
    grads_val = features_grad.numpy()[0]

    # 池化梯度
    weights = np.mean(grads_val, axis=(1, 2))

    # 生成类激活映射
    cam = np.zeros(features.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * features[0][i].numpy()

    # 应用ReLU
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(cam)
        plt.show()

    # 将CAM叠加到原始图像上
    img = cv2.imread(img_path)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # 叠加到原始图像
    superimposed_img = cam * 0.4 + img
    cv2.imwrite(save_path, superimposed_img)


if __name__ == '__main__':
    # model_re2 = Res2Net_vd(layers=50, scales=4, width=26, class_dim=4)
    ResNetTweaksTSMModify = ResNetTweaksTSMModify(depth=50, num_seg=8)

    ResNetTweaksTSMModify_state_dict = paddle.load(
        "/mnt/sdb1/swf/project/PaddleVideo/output/PPTSM_RESULTS/add_C2f_in_backbone_and_MultiDilatelocalAttention_in_head/ppTSM_task_VideoClassfy_frames_dense_no_pretrained_modify_best.pdparams")
    ResNetTweaksTSMModify.set_state_dict(ResNetTweaksTSMModify_state_dict, use_structured_name=True)
    use_gpu = True

    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    ResNetTweaksTSMModify.eval()

    draw_CAM(ResNetTweaksTSMModify,
             '/mnt/sdb1/swf/data/dataset_last/task_VideoClassfy_frames/foreign_matter/7/img_00023.jpg', 'test1.jpg',
             transform=None, visual_heatmap=True)
