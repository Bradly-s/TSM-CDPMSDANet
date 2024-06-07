import os

# 图片所在目录路径
img_dir = '/mnt/sdb1/swf/data/UCF24/rgb-images/'

# 遍历图片文件，重命名并加上前缀
for filename in os.listdir(img_dir):

    file_path = os.path.join(img_dir, filename)
    for file in os.listdir(file_path):
        file_ = os.path.join(file_path, file)
        for this_file in os.listdir(file_):
            # print(this_file)
            if this_file.endswith('.jpg'):
                new_filename = 'img_' + this_file
                os.rename(os.path.join(file_, this_file), os.path.join(file_, new_filename))
print("done!")