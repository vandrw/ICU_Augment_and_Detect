import tensorflow as tf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from categorization.data_utils import load_data
from categorization.models import *

def get_accuracy(test_labels, prediction_labels, thresh=0.5):
    sum_acc = 0.0
    for i in range(len(test_labels)):
        if (test_labels[i] == (prediction_labels[i] >= thresh)):
            sum_acc += 1
    
    return sum_acc / len(test_labels)

print("Loading data...")

image_size = 128
thresholds = [0.5, 0.6, 0.7, 0.8]
folds = 10

test_faces, _ = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "face")
test_images_mouth, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "mouth")
test_images_face, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "nose")
test_images_skin, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "skin")
test_images_right_eye, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "_right")

test_images = [test_images_mouth, test_images_face, test_images_skin, test_images_right_eye]

print("Loading model and making predictions...")

for feature in ["mouth", "nose", "skin", "eye", "stacked"]:
    print("Predicting for " + feature + "...")
    accs = {
        0.5: [],
        0.6: [],
        0.7: [],
        0.8: []
        }
    for fold_no in range(1,folds+1):
        model = tf.keras.models.load_model(
            "categorization/model_saves/" + feature + "/model_" + str(fold_no) + '.h5', compile=False)

        if feature == "stacked":
            imgs = test_images
        elif feature == "mouth":
            imgs = test_images[0]
        elif feature == "nose":
            imgs = test_images[1]
        elif feature == "skin":
            imgs = test_images[2]
        elif feature == "eyes":
            imgs = test_images[3]

        pred = model.predict(imgs)

        for thresh in thresholds:
            acc = get_accuracy(test_labels, pred, thresh)
            # print("[Threshold {:.2f}] Accuracy fold {:d}: {:.4f}".format(thresh, fold_no, acc))
            accs[thresh].append(acc)

    # for thresh in thresholds:
    #     print("[{}] Mean accuracy on {} folds (threshold={:.2f}): {:.4f}".format(feature.upper(), folds, thresh, accs[thresh]/folds))
    for thresh in thresholds:
        print(thresh, accs[thresh])
    print("---------------------------------------------------\n")
