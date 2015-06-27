from __future__ import print_function
import os

def get_filepaths(directory):
    f = open('./object_pose_recognition/data/image_file_list.txt', 'w');

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if ".png" not in filepath: continue
            print(filepath + " 1", file=f);

full_file_paths = get_filepaths("./object_pose_recognition/data/OPR-Dataset-New/")
