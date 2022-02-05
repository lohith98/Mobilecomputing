import json
import numpy as np
import pandas as pd
import os
import configparser
from os.path import join, dirname, isfile, basename
from configparser import ConfigParser
import pathlib


current_dir = dirname(os.path.realpath(__file__))
config = ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(join(current_dir, '..', '..', 'properties.ini'))


def convert(posenet_json):

    print('Converting - ', posenet_json)
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']

    data = json.loads(open(posenet_json, 'r').read())
    csv_data = np.zeros((len(data), len(columns)))
  #  output_path = join(pathlib.Path(posenet_json).parent.absolute(), 'key_points.csv')

    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)

    pd.DataFrame(csv_data, columns=columns).to_csv(output_path, index_label= '#Frames')
    print('Converted - ', output_path)


if __name__ == '__main__':

    files = os.listdir('path_to_posenets')
    for file in files:
        path = join('path_to_posenets', file)
        if isfile(path) and 'key_points' in str(file):
            convert(path)
