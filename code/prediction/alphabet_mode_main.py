from handshape_feature_extractor import HandShapeFeatureExtractor
from os.path import join
import glob
import cv2
import numpy as np
import os
import tensorflow as tf


def get_inference_vector_one_frame_alphabet(frames):
    """ Ref:  """
    model = HandShapeFeatureExtractor.get_instance()
    vectors = []
    for frame in frames:

        try:
            image = cv2.imread(frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            results = model.extract_feature(image)
            results = np.squeeze(results)
            predicted = np.where(results == max(results))[0][0]
            vectors.append(predicted)

        except Exception as e:
            # skipped erroneous frame
            continue

    return vectors


def load_labels(label_file):
    """ Loads labels """
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())

    return label


def load_label_dicts(labels):
    """ Return dict for labels to ID and vice-versa """
    id_to_labels = load_labels(labels)
    labels_to_id = {}
    i = 0
    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id


def predict_alphabets_from_frames(config, path):

    print('Predicting alphabet from frames in - ', path)
    path = join(path, '*.png')
    frames = glob.glob(path)
    frames.sort()

    prediction_vectors = get_inference_vector_one_frame_alphabet(frames)

    if prediction_vectors is None:
        return '?'

    id_to_labels, labels_to_id = load_label_dicts(config['DEFAULT']['labels_path'])

    predictions = []
    for vector in prediction_vectors:
        for label in labels_to_id:
            if vector == labels_to_id[label]:
                predictions.append(label)

    return predictions


def predict_words_from_frames(config, path, till):

    print('Predicting alphabet from frames in - ', path)
    path = os.path.join(path, "*.png")
    frames = glob.glob(path)
    frames.sort()
    files = frames[:till]

    prediction_vector = get_inference_vector_one_frame_alphabet(files)

    if prediction_vector is None:
        return '?'

    id_to_labels, labels_to_id = load_label_dicts(config['DEFAULT']['labels_path'])

    predictions = []

    for i in range(len(prediction_vector)):
        for ins in labels_to_id:
            if prediction_vector[i] == labels_to_id[ins]:
                predictions.append(ins)

    return predictions
