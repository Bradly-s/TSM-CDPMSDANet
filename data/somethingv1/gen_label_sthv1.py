# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1
# https://github.com/mit-han-lab/temporal-shift-module/blob/master/tools/gen_label_sthv1.py

import os

if __name__ == '__main__':
    path = "/mnt/sdb1/swf/data/something-something-v1/"
    dataset_name = 'something-something-v1'  # 'jester-v1'
    with open(path + '%s-labels.csv' % dataset_name) as f:
        lines = f.readlines()
        # print("lines:", lines)
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open(path + 'category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [path + '%s-validation.csv' % dataset_name, path + '%s-train.csv' % dataset_name]
    files_output = [path + 'val_videofolder.txt', path + 'train_videofolder.txt']
    print("files_input:", files_input)
    print("====================================================================")
    print("files_output:", files_output)

    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
        output = []
        print("folders[0]：", folders[0])    # 85, 100218
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            print("curFolder:", curFolder)
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('/mnt/sdb1/swf/data/something-something-v1/something-something-v1', curFolder))
            # output.append('%s %d %d' % ('something/v1/img/' + curFolder, len(dir_files), curIDX))
            output.append('%s %d %d' % ('/mnt/sdb1/swf/data/something-something-v1/something-something-v1/' + curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))