import tensorflow as tf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from categorization.cnn import load_data

def get_accuracy(test_labels, prediction_labels):
    sum_acc = 0.0
    for i in range(len(test_labels)):
        if (test_labels[i] == (prediction_labels[i] >= 0.5)):
            sum_acc += 1
    
    return sum_acc / len(test_labels)

print("Loading data...")

image_size = 128
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

for feature in ["mouth", "nose", "skin", "eyes", "stacked"]:
    print("Predicting for " + feature + "...")
    model = tf.keras.models.load_model(
        "categorization/model_saves/" + feature + "/model.h5")

    if feature == "stacked":
        pred = model.predict(test_images)
        print("Accuracy: ", get_accuracy(test_labels, pred))
        plt.figure(figsize=(10, 10))
        for i in range(30):
            plt.subplot(6, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_faces[i], cmap=plt.cm.binary)
            result = pred[i]
            real = test_labels[i]
            plt.xlabel("%d (%.2f), real: %.2f" % ((result >= 0.5), result, real))
        plt.suptitle("Results " + feature + " model")
        plt.savefig("data/plots/predictions.png")
        continue

    elif feature == "mouth":
        imgs = test_images[0]
    elif feature == "nose":
        imgs = test_images[1]
    elif feature == "skin":
        imgs = test_images[2]
    elif feature == "eyes":
        imgs = test_images[3]

    pred = model.predict(imgs)
    print("Accuracy: ", get_accuracy(test_labels, pred))

    plt.figure(figsize=(10, 10))
    plt.title("Results " + feature + " model")
    for i in range(30):
        plt.subplot(6, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        result = pred[i]
        real = test_labels[i]
        plt.xlabel("%d (%.2f), real: %.2f" % ((result >= 0.5), result, real))
    plt.suptitle("Results " + feature + " model")
    plt.savefig("data/plots/predictions_" + feature + ".png")
