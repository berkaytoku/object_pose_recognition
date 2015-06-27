from __future__ import print_function
import os


def get_filepaths(directory):
    f = open('./object_pose_recognition/data/desc_image_files.txt', 'w');
    classNums = {'CableBoxGreenCluttered':1,'HouseCluttered':2,'InstallationCapOrangeCluttered':3,'PowerSocketRedInsideCluttered':4,'VolvoPoliceWhiteCluttered':5}
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if ".png" not in filepath: continue
            if "test" in filepath: continue
            words = filename.split("_");
            if(len(words) > 5 and classNums.get(words[0])):
                curve = int(words[3])
                rot = int(words[5].split(".")[0])
            print(filepath + " " + str(classNums.get(words[0])) + " " + str(curve) + " " + str(rot), file=f);


full_file_paths = get_filepaths("./object_pose_recognition/data/OPR-Dataset-New")