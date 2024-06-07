import os

# 图片所在目录路径
img_dir = '/mnt/sdb1/swf/data/something-something-v1/something-something-v1/'

# 遍历图片文件，重命名并加上前缀
for filename in os.listdir(img_dir):

    file_path = os.path.join(img_dir, filename)
    # print("file_path:", file_path)
    for file in os.listdir(file_path):
        if file.endswith('.jpg'):
            new_filename = 'img_' + file
            os.rename(os.path.join(file_path, file), os.path.join(file_path, new_filename))
print("done!")
