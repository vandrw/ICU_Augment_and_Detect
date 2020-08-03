import tensorflow as tf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from categorization.cnn import load_data

print("Loading data...")

image_size = 128
test_images_mouth, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "mouth")
test_images_face, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "face")
test_images_skin, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "skin")
test_images_right_eye, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "right")

perm = np.random.permutation(len(test_images_mouth))
test_images = [test_images_mouth[perm], test_images_face[perm],
               test_images_skin[perm], test_images_right_eye[perm]]
test_labels = test_labels[perm]

print("Loading model and making predictions...")

for feature in ["mouth", "face", "skin", "eyes", "stacked"]:
    print("Predicting for " + feature + "...")
    model = tf.keras.models.load_model(
        "categorization/model_saves/" + feature + "/model.h5")

    if feature == "stacked":
        pred = model.predict(test_images)
        plt.figure(figsize=(10, 10))
        for i in range(30):
            plt.subplot(6, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_images[1][i], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            result = pred[i].argmax()
            real = test_labels[i].argmax()
            plt.xlabel("%d (%.3f), real: %d" % (result, pred[i][result], real))
        plt.suptitle("Results " + feature + " model")
        plt.savefig("data/plots/predictions.png")
        continue

    elif feature == "mouth":
        imgs = test_images[0]
    elif feature == "face":
        imgs = test_images[1]
    elif feature == "skin":
        imgs = test_images[2]
    elif feature == "eyes":
        imgs = test_images[3]
    
    pred = model.predict(imgs)

    plt.figure(figsize=(10, 10))
    plt.title("Results " + feature + " model")
    for i in range(30):
        plt.subplot(6, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        result = pred[i].argmax()
        real = test_labels[i].argmax()
        plt.xlabel("%d (%.3f), real: %d" % (result, pred[i][result], real))
    plt.suptitle("Results " + feature + " model")
    plt.savefig("data/plots/predictions_" + feature + ".png")
