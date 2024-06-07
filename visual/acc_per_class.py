import pandas as pd
import matplotlib.pyplot as plt
import os



def make_acc_comparison(class_file_path, ucf24_excel, result, datasett):

    # 初始化一个列表来存储文件内容
    classids = []

    # 使用with语句安全打开文件
    with open(class_file_path, 'r') as file:
        # 逐行读取文件
        for line in file:
            # 将每一行添加到列表中，你可以根据需要处理每行数据
            classids.append(line.strip())  # strip() 方法移除行尾的换行符


    # 确保保存路径存在
    os.makedirs(result, exist_ok=True)

    # 读取Excel文件
    df = pd.read_excel(ucf24_excel, header=None)

    # 第一行为横坐标
    x_axis = df.iloc[0].values
    x_axis = [i.split(" ")[1] for i in x_axis]
    x_axis = [classids[int(i)] for i in x_axis]

    x_axis = [i.split(" ")[1] for i in x_axis]
    print("x_axis:", x_axis)

    # 第二行为数据1的纵坐标
    data1 = df.iloc[1].values
    # print("data1:", data1)

    # 第三行为数据2的纵坐标
    data2 = df.iloc[2].values
    # 转换 data2 中的所有字符串为浮点数
    data2 = [float(i) for i in data2]
    # print("data2:", data2)

    # 绘制折线图
    # plt.figure(figsize=(10, 5))
    plt.figure(figsize=(17, 5))

    plt.plot(x_axis, data1, label='PPTSM', color='blue', marker='o')  # 数据1的折线图，颜色为蓝色
    plt.plot(x_axis, data2, label='ours', color='red', marker='x')  # 数据2的折线图，颜色为红色

    # 添加图例
    plt.legend()

    # 设置标题和坐标轴标签
    plt.title(datasett + ' Class Accuracies Comparison')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')

    # 倾斜横坐标轴的文字
    plt.xticks(rotation=75)  # 旋转45度，或者你可以选择你想要的角度
    # 调整子图参数，特别是底部的边距，以适应更高的刻度标签
    plt.subplots_adjust(bottom=0.29)  # 增加底部的边距

    # 显示网格
    plt.grid(True)

    # 图片保存的完整路径
    image_path = os.path.join(result, datasett + '_class_accuracies_comparison.png')

    # 保存图形
    plt.savefig(image_path)

    # 显示图形
    plt.show()

if __name__ == '__main__':
    class_file_path = "/mnt/sdb1/swf/data/hmdb51/annotations/classInd.txt"

    datasett = "hmdb51"

    # 折线图的保存路径
    result = "/mnt/sdb1/swf/project/PaddleVideo/output/PPTSM_RESULTS/visul/" + datasett + "/"
    # Excel文件路径
    ucf24_excel = "/mnt/sdb1/swf/project/PaddleVideo/output/PPTSM_RESULTS/visul/" + datasett + "/class_accuracies.xlsx"

    make_acc_comparison(class_file_path, ucf24_excel, result, datasett)

