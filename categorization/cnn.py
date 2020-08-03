'''
Plots:
- accuracy
- precision
- recall
- ROC
- binary classification
- explanation part

Other:
- plot with what we did (extract, neural network, train)
'''
from augment.face_org import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
import pickle
import sklearn.metrics
import random

sys.path.append(os.getcwd())


def load_data(folder_sick, folder_healthy, image_size, ftype):
    files_healthy = os.listdir(folder_healthy)
    files_sick = os.listdir(folder_sick)
    data = []
    labels = []
    for filename in files_healthy:
        sick = np.array([0, 1])
        full_path = folder_healthy + "/" + str(filename)
        if ftype in filename and os.path.isfile(full_path) and "n2" not in filename:
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(
                image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype=np.int32))
            labels.append(np.asarray(sick, dtype=np.int32))
    for filename in files_sick:
        sick = np.array([1, 0])
        full_path = folder_sick + "/" + str(filename)
        if ftype in filename and os.path.isfile(full_path):
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(
                image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype=np.int32))
            labels.append(np.asarray(sick, dtype=np.int32))
    return np.asarray(data, dtype=np.float64) / 255, np.asarray(labels, dtype=np.int32)


def load_shuffled_data(folder_sick, folder_healthy, image_size, ftype):
    data, labels = load_data(folder_sick, folder_healthy, image_size, ftype)
    permutation = np.random.permutation(len(data))
    return data[permutation], labels[permutation]


def make_model(image_size, feature):
    model = models.Sequential()

    model.add(layers.Conv2D(image_size, (3, 3), padding="same", activation='relu',
                            input_shape=(image_size, image_size, 3),
                            name="input_" + str(feature)))

    model.add(layers.BatchNormalization(name="batch1_" + str(feature)))
    model.add(layers.Conv2D(int(image_size / 2), (3, 3),
                            activation='relu', name="conv1_" + str(feature)))
    model.add(layers.BatchNormalization(name="batch2_" + str(feature)))
    model.add(layers.MaxPooling2D((2, 2), name="max1_" + str(feature)))

    model.add(layers.Conv2D(int(image_size/4), (3, 3),
                            activation='relu', name="conv2_" + str(feature)))
    model.add(layers.BatchNormalization(name="batch3_" + str(feature)))
    model.add(layers.MaxPooling2D((2, 2), name="max2_" + str(feature)))

    model.add(layers.Conv2D(int(image_size/8), (3, 3),
                            activation='relu', name="conv5_" + str(feature)))
    model.add(layers.BatchNormalization(name="batch6_" + str(feature)))
    model.add(layers.MaxPooling2D((2, 2), name="max3_" + str(feature)))

    model.add(layers.Conv2D(int(image_size/16), (3, 3),
                            activation='relu', name="conv6_" + str(feature)))
    model.add(layers.BatchNormalization(name="batch7_" + str(feature)))
    model.add(layers.AveragePooling2D((2, 2), name="avg1_" + str(feature)))

    model.add(layers.Flatten(name="flatten_" + str(feature)))
    model.add(layers.Dense(48, activation='relu',
                           name="dense1_" + str(feature)))
    model.add(layers.Dropout(0.3, name="dropout1_" + str(feature)))

    model.add(layers.Dense(16, activation='relu',
                           name="dense2_" + str(feature)))
    model.add(layers.Dropout(0.5, name="dropout2_" + str(feature)))

    model.add(layers.Dense(2, activation='softmax',
                           name="dense3_" + str(feature)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    return model


def load_data_eyes(image_folder_sick, image_folder_healthy, image_size):
    images_left, labels_left = load_shuffled_data(
        image_folder_sick, image_folder_healthy, image_size, "_left")
    images_right, labels_right = load_shuffled_data(
        image_folder_sick, image_folder_healthy, image_size, "_right")

    images = np.concatenate((images_left, images_right), axis=0)
    labels = np.concatenate((labels_left, labels_right), axis=0)

    permutation = np.random.permutation(len(images))

    return images[permutation], labels[permutation]


def save_history(save_path, history, feature):
    with open(save_path + str(feature) + "/history.pickle", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def plot_roc(feature, saved_model, test_images, test_labels):
    pred = saved_model.predict(test_images)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(
        test_labels.argmax(axis=1), pred.argmax(axis=1))
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("data/plots/" + str(feature) + "_auc.png")
    plt.figure()


def plot_acc(feature, history):
    plt.plot(history.history['accuracy'], label="Training accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation accuracy")
    plt.legend()
    plt.ylim((0.3, 1.05))
    plt.xlim((0, len(history.history["accuracy"])))
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.title("Accuracy of the " + str(feature) + " CNN")
    plt.savefig("data/plots/" + str(feature) + "_accuracy.png")
    plt.figure()


if __name__ == "__main__":

    image_folder_sick = 'data/parsed/brightened/sick'
    image_folder_healthy = 'data/parsed/brightened/healthy'
    image_folder_val_sick = 'data/parsed/validation_sick'
    image_folder_val_healthy = 'data/parsed/validation_healthy'
    save_path = 'categorization/model_saves/'
    image_size = 128
    face_features = ["mouth", "face", "skin", "eyes"]

    for feature in face_features:

        print("[INFO] Training %s" % (feature))

        if feature == "eyes":
            test_images, test_labels = load_data_eyes(
                image_folder_val_sick, image_folder_val_healthy, image_size)
            train_images, train_labels = load_data_eyes(
                image_folder_sick, image_folder_healthy, image_size)

        else:
            test_images, test_labels = load_shuffled_data(
                image_folder_val_sick, image_folder_val_healthy, image_size, feature)
            train_images, train_labels = load_shuffled_data(
                image_folder_sick, image_folder_healthy, image_size, feature)

        model = make_model(image_size, feature)

        monitor = "val_accuracy"

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode='max', patience=10, verbose=1)
        model_check = tf.keras.callbacks.ModelCheckpoint(
            save_path + str(feature) + '/model.h5', monitor=monitor, mode='max', verbose=1, save_best_only=True)

        history = model.fit(train_images, train_labels, epochs=50,
                            batch_size=1, callbacks=[early_stopping, model_check], validation_data=(test_images, test_labels))

        save_history(save_path, history, feature)

        saved_model = tf.keras.models.load_model(
            save_path + str(feature) + '/model.h5')

        plot_roc(feature, saved_model, test_images, test_labels)
        plot_acc(feature, history)
