import os
import subprocess
import math
import pandas as pd
import configparser
from alphabet_mode_main import predict_alphabets_from_frames, predict_words_from_frames
from os.path import join, dirname, basename
from pandas import DataFrame
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter
from frames_extractor import extract_frames
from convert_to_csv import convert

current_dir = dirname(os.path.realpath(__file__))
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(join(current_dir, '..', '..', 'properties.ini'))


def setup():
    """ Creates required directories """
    if not os.path.exists(config['DEFAULT']['tmp_frames_dir']):
        os.makedirs(config['DEFAULT']['tmp_frames_dir'])
    if not os.path.exists(config['DEFAULT']['frames_dir']):
        os.makedirs(config['DEFAULT']['frames_dir'])
    if not os.path.exists(config['DEFAULT']['output_dir']):
        os.makedirs(config['DEFAULT']['output_dir'])


def get_frames(source, method=1):
    """ Extracts frames from either frame_extractor file(1) or handtracking library(2) """
    frames_path = None
    if method == 1:
        frames_path = extract_frames(config, source)
    else:
        print(f"{config['DEFAULT']['yolo_script']}")
        cmd = subprocess.Popen(config['DEFAULT']['yolo_script'], shell = True, stdout = subprocess.PIPE)
        cmd.wait()
        crop_frames()
        filename = basename(source).split('.')[0]
        frames_path = join(config['DEFAULT']['frames_dir'], filename)

    frames = [frame for frame in os.listdir(frames_path)]
    print('Total frames generated- ', len(frames))

    return frames_path


def get_keypoints(frames_dir):
    print("Trying to get key points")
    print(f"{config['WORD_RECOGNITION']['poseest_cmd']}")
    cmd = subprocess.Popen(config['WORD_RECOGNITION']['poseest_cmd'], shell = True, stdout = subprocess.PIPE)
    cmd.wait()
    convert(os.path.join(frames_dir, 'key_points.json'))
    keypoints = pd.read_csv(join(frames_dir, 'key_points.csv'))

    return keypoints


def crop_frames():
    print(f"{config['DEFAULT']['handtracking_script']}")
    cmd = subprocess.Popen(config['DEFAULT']['handtracking_script'], shell = True, stdout = subprocess.PIPE)
    cmd.wait()

    return True


def get_final_prediction(predictions):
    """ Returns the most common label from video frames as the final prediction """
    if not predictions or len(predictions) == 0:
        return '?'

    final_prediction = Counter(predictions).most_common(1)[0][0]

    return final_prediction


def save(predictions, option):
    """ Saves predictions and metrics in result and report files """
    results_path = join(config['DEFAULT']['output_dir'], 'result_' + option + '.csv')
    report_path = join(config['DEFAULT']['output_dir'], 'report_' + option + '.txt')

    predictionsDf = DataFrame(predictions, columns = ['actual', 'prediction'])
    predictionsDf.to_csv(results_path, index = False)

    y_true = predictionsDf.actual.tolist()
    y_pred = predictionsDf.prediction.tolist()
    report = classification_report(y_true, y_pred, digits = 2, zero_division = 1)
    with open(report_path, "w") as writer:
        f1 = f1_score(y_true, y_pred, zero_division = 1, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)

        print("RESULTS:\n")
        print("%s" % report)
        print("F1 score: %f" % f1)
        print("Accuracy: %f" % accuracy)

        writer.write("F1 score:\n")
        writer.write(str(f1))
        writer.write("\n\nAccuracy:\n")
        writer.write(str(accuracy))
        writer.write("\n\n")
        writer.write(report)


def predict_alphabet():
    """ Predicts alphabet """
    videos = os.listdir(config['ALPHABET_RECOGNITION']['alphabets_dir'])
    if not videos or len(videos) == 0:
        print('No videos available for demo.')
        return
    setup()
    predictions = []
    sep = "*" * 100

    for video in videos:

        print(sep)
        if not video.endswith('.mp4'):
            print(f'{video} is not an mp4 file. Skipping.')
            continue
        print('Processing video - ', video)
        ground_truth = video[0]

        path = join(config['ALPHABET_RECOGNITION']['alphabets_dir'], video)
        print(path)
        video_frames_dir = get_frames(path)
        print(video_frames_dir)
        keypoints = get_keypoints(video_frames_dir)

        # predict
        prediction_frames = predict_alphabets_from_frames(config, 'frames_dir')
        # print('Frame-wise predictions - ', prediction)
        prediction = get_final_prediction(prediction_frames)

        print(f'\nActual: {ground_truth}, Prediction: {prediction}')
        predictions.append([ground_truth, prediction])

    print(sep)
    save(predictions, 'alphabets')


def predict_word():
    """ Predicts word """
    videos = os.listdir(config['WORD_RECOGNITION']['words_dir'])
    if not videos or len(videos) == 0:
        print('No videos available for demo.')
        return
    setup()
    predictions = []
    sep = "*" * 50

    for video in videos:

        print(sep)
        if not video.endswith('.mp4'):
            print(f'{video} is not an mp4 file. Skipping.')
            continue
        print('Processing video - ', video)
        ground_truth = video.split('.')[0]

        path = join(config['WORD_RECOGNITION']['words_dir'], video)

        video_frames_dir = get_frames(config,path,2)

        keypoints = get_keypoints(video_frames_dir)

        right_x = keypoints.rightWrist_x
        right_y = keypoints.rightWrist_y
        left_x = keypoints.leftWrist_x
        left_y = keypoints.leftWrist_y

        letters = []
        i = 0
        threshold = 0.5

        while i < keypoints.shape[0]:
            # calculate euclidean distance between the wrist(left & right) location of successive frames
            while i < keypoints.shape[0] - 1 \
                    and math.sqrt(((right_x[i + 1] - right_x[i]) ** 2) + ((right_y[i + 1] - right_y[i]) ** 2)) < threshold\
                    and math.sqrt(((left_x[i + 1] - left_x[i]) ** 2) + ((left_y[i + 1] - left_y[i]) ** 2)) < threshold:
                i += 1

            if i == keypoints.shape[0]:
                print('No frame selected.')
                break
            print(f'Selected #frame - ', i)

            # predict
            prediction_frames = predict_words_from_frames(config, 'frames_dir', i)
            prediction = get_final_prediction(prediction_frames)

            letters.append(prediction)
            i += 1

        word = ''.join(letters).lower()
        print(f'\nActual: {ground_truth}, Prediction: {word}')
        predictions.append([ground_truth, word])

    print(sep)
    save(predictions, 'words')


def process(option):
    """ Executes tasks according to user input """
    if option == 1:
        predict_alphabet()
    elif option == 2:
        predict_word()

    print('Successfully completed!')


if __name__ == '__main__':

    """ User input for task option """
    print('Instructions: The videos have to be present in the demo/ folder for prediction')
    print('Select Task: \n1. Alphabet Recognition \n2. Word Recognition')
    option = int(input('Enter option 1 or 2\n'))

    process(option)
