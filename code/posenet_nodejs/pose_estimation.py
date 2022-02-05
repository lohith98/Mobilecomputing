import os
import cv2 as cv
from os.path import join
import numpy as np
from PIL import Image
from pose_engine import PoseEngine


def get_pose_estimation(path):

    folders = [folder for folder in os.listdir(path)]
    for folder in folders:

        output_path = join(path, folder, 'key_points.json')
        pose_list = []
        for frame in os.listdir(join(path, folder)):

            image = cv.imread(join(path, folder, frame))
            pose = None
            pose_list.append(pose)

        print(pose_list)

    return folders


if __name__ == "__main__":

    path = "C:/Users/Ram/Desktop/asl_fingerspelling/demo/frames"
    print(get_pose_estimation(path))