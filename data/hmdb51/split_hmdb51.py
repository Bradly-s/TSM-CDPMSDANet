import os
import shutil
import random

def generate_list(dataset_path):
    class_folders = sorted(os.listdir(dataset_path))
    for (i, class_folder) in enumerate(class_folders):
        print("i:", i, ",class_folder:", class_folder)
        with open("/mnt/sdb1/swf/data/hmdb51/annotations/classInd.txt", "a") as file:
            file.write(str(i + 1) + " " + class_folder + "\n")

def split_dataset(dataset_path, train_ratio=0.7, val_ratio=0.3):
    class_folders = sorted(os.listdir(dataset_path))

    for class_folder in class_folders:
        # print("class_folder:", class_folder)
        class_path = os.path.join(dataset_path, class_folder)
        video_folders = sorted(os.listdir(class_path))

        num_videos = len(video_folders)
        print("num_videos:", num_videos)
        num_train = int(num_videos * train_ratio)
        num_val = num_videos - num_train
        print("num_train:", num_train, ",--num_val:", num_val)

        random.shuffle(video_folders)

        train_videos = video_folders[:num_train]
        val_videos = video_folders[num_train:]

        for video_folder in train_videos:
            src_path = os.path.join(class_path, video_folder)
            dest_path = os.path.join("/mnt/sdb1/swf/data/hmdb51/hmdb51_frames_split/train", class_folder, video_folder)
            print("train_src_path：", src_path)
            print("train_dest_path：", dest_path)
            # shutil.move(src_path, dest_path)

        for video_folder in val_videos:
            src_path = os.path.join(class_path, video_folder)
            dest_path = os.path.join("/mnt/sdb1/swf/data/hmdb51/hmdb51_frames_split/val", class_folder, video_folder)
            print("val_src_path：", src_path)
            print("val_dest_path：", dest_path)
            # shutil.move(src_path, dest_path)


def generate_trainvallist01(dataset_path, train_ratio=0.7, val_ratio=0.3):
    class_folders = sorted(os.listdir(dataset_path))

    for class_folder in class_folders:
        # print("class_folder:", class_folder)
        class_path = os.path.join(dataset_path, class_folder)
        video_folders = sorted(os.listdir(class_path))

        num_videos = len(video_folders)
        print("num_videos:", num_videos)
        num_train = int(num_videos * train_ratio)
        num_val = num_videos - num_train
        print("num_train:", num_train, ",--num_val:", num_val)

        random.shuffle(video_folders)

        train_videos = video_folders[:num_train]
        val_videos = video_folders[num_train:]

        # print("val_videos:", val_videos)
        print("===================================================================================")

        for video_folder in train_videos:
            train_video_path = os.path.join(class_folder, video_folder)
            # print("val_video_path:", val_video_path)
            with open("/mnt/sdb1/swf/data/hmdb51/annotations/trainlist01.txt", "a") as file:
                file.write(train_video_path + "\n")

        for video_folder in val_videos:
            val_video_path = os.path.join(class_folder, video_folder)
            # print("val_video_path:", val_video_path)
            with open("/mnt/sdb1/swf/data/hmdb51/annotations/vallist01.txt", "a") as file:
                file.write(val_video_path + "\n")


if __name__ == "__main__":
    dataset_path = "/mnt/sdb1/swf/data/hmdb51/hmdb51_frames"
    train_ratio = 0.7
    val_ratio = 0.3

    # Create train, val, and test directories
    os.makedirs("train", exist_ok=True)
    os.makedirs("val", exist_ok=True)

    generate_list(dataset_path)

    # split_dataset(dataset_path, train_ratio, val_ratio)
    # generate_trainvallist01(dataset_path, train_ratio, val_ratio)


