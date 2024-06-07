import os
import glob
import random
import fnmatch
import re


# def get_list(path, class_id, key_func=lambda x: x[-11:], rgb_prefix='img_', level=1):
def get_list(path, class_id, key_func=lambda x: x[-11:], rgb_prefix='img_', level=1):
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
        print("frame_folders:" + str(frame_folders))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
        print("frame_folders:" + str(frame_folders))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory):
        lst = os.listdir(directory)
        cnt = len(fnmatch.filter(lst, rgb_prefix + '*'))
        return cnt

    # check RGB
    video_dict = {}
    for f in frame_folders:
        cnt = count_files(f)
        print("cnt:" + str(cnt))
        print("f:" + f)
        k = key_func(f)
        if level == 2:
            k = k.split("/")[0]
            # print("k:" + k)
            # 注：此处可能要根据文件夹调整（当出现KeyError时）
            # k = k.split("\\")[-2]
            # print("k.split('\\')[-2]:" + k)

        video_dict[f] = str(cnt) + " " + str(class_id[k])
    return video_dict


def fight_splits(video_dict, train_list_path, val_list_path, train_percent=0.8):
    videos = list(video_dict.keys())

    train_num = int(len(videos) * train_percent)

    train_list = []
    val_list = []

    random.shuffle(videos)

    for i in range(train_num):
        train_list.append(videos[i] + " " + str(video_dict[videos[i]]))
    for i in range(train_num, len(videos)):
        val_list.append(videos[i] + " " + str(video_dict[videos[i]]))

    print("train:", len(train_list), ",val:", len(val_list))

    with open(train_list_path, "w") as f:
        for item in train_list:
            f.write(item + "\n")

    with open(val_list_path, "w") as f:
        for item in val_list:
            f.write(item + "\n")


def split_data(frame_dir, train_percent, class_id, train_list_path, val_list_path):
    # frame_dir = "/media/dp504/36675aac-be99-45b0-abf0-9c74c037c1b7/PycharmProjects/swf/PaddleVideo/data/five_dataset_small/fight_rawframes"
    level = 2
    train_percent = 0.8

    if level == 2:
        def key_func(x):
            return '/'.join(x.split('/')[-2:])
    else:
        def key_func(x):
            return x.split('/')[-1]

    video_dict = get_list(frame_dir, class_id, key_func=key_func, level=level)
    print("number:", len(video_dict))

    fight_splits(video_dict, train_list_path, val_list_path, train_percent)


if __name__ == '__main__':
    # frame_dir = "D:/Datasets/task_VideoClassfy_frames"
    frame_dir = "/media/dp504/36675aac-be99-45b0-abf0-9c74c037c1b7/PycharmProjects/swf/datasset/dataset_last/task_VideoClassfy_frames"
    train_percent = 0.8
    class_id = {
        "foreign_matter": 1,
        "jump": 2,
        "kick": 3,
        "pick_door": 4
    }


    # train_list_path = frame_dir + str(list(class_id.keys())[1]) + "_train_list.txt"
    # print("train_list_path:" + train_list_path)
    # val_list_path = frame_dir + str(list(class_id.keys())[1]) + "_val_list.txt"
    # print("val_list_path:" + val_list_path)
    train_list_path = frame_dir + "_train_list.txt"
    print("train_list_path:" + train_list_path)
    val_list_path = frame_dir + "_val_list.txt"
    print("val_list_path:" + val_list_path)

    split_data(frame_dir, train_percent, class_id, train_list_path, val_list_path)
