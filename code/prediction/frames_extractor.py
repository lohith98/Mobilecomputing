import cv2
import os
from os.path import join, basename


def extract_frames(config, path):

    filename = basename(path)
    print('Extracting frames from - ', path)
    video = cv2.VideoCapture(path)
    flip = True
    count = 0
    success = 1
    arr_img = []

    # If such a directory doesn't exist, creates one and stores its Images
    frames_path = join(config['DEFAULT']['tmp_frames_dir'], filename.split('.')[0])
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)

        while success:

            success, image = video.read()
            # Frames when generated are getting rotated clockwise by above method, so correcting it
            if flip:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            arr_img.append(image)
            count += 1

        # Sub sampling the number of frames
        # numbers = sorted(random.sample(range(len(arr_img)), 45))
        count = 0
        for img in arr_img:

            # final_img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            final_img = img
            cv2.imwrite(frames_path + "/%d.png" % count, final_img)
            count += 1

    return frames_path
