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

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
import pickle

import random

sys.path.append(os.getcwd())
from augment.face_org import *

def load_data(folder_sick, folder_healthy, image_size, type):
    files_healthy = os.listdir(folder_healthy)
    files_sick = os.listdir(folder_sick)
    data = []
    labels = []
    for filename in files_healthy:
        sick = 0
        full_path = folder_healthy + "/" + str(filename)
        if type in filename and os.path.isfile(full_path) and "n2" not in filename:
            image =  cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype = np.int32))
            labels.append(np.asarray(sick, dtype = np.int32))
    for filename in files_sick:
        sick = 1
        full_path = folder_sick + "/" + str(filename)
        if type in filename and os.path.isfile(full_path):
            image =  cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype = np.int32))
            labels.append(np.asarray(sick, dtype = np.int32))
    return np.asarray(data, dtype=np.int32) / 255, np.asarray(labels, dtype=np.int32)

def load_shuffled_data(folder_sick, folder_healthy, image_size, type):
    data, labels = load_data(folder_sick, folder_healthy, image_size, type)
    permutation = np.random.permutation(len(data))
    return data[permutation], labels[permutation]


def make_model(image_size, feature):
    model = models.Sequential()

    model.add(layers.Conv2D(image_size, (3, 3), padding="same", activation='relu', 
                            input_shape=(image_size, image_size, 3), 
                            name = "input_" + str(feature)))

    model.add(layers.BatchNormalization(name = "batch1_" + str(feature)))
    model.add(layers.Conv2D(int(image_size / 2), (3, 3), activation='relu', name = "conv1_" + str(feature)))
    model.add(layers.BatchNormalization(name = "batch2_" + str(feature)))
    model.add(layers.MaxPooling2D((2, 2), name = "max1_" + str(feature)))

    model.add(layers.Conv2D(int(image_size/4), (3, 3), activation='relu', name = "conv2_" + str(feature)))
    model.add(layers.BatchNormalization(name = "batch3_" + str(feature)))
    # model.add(layers.Conv2D(int(image_size/2), (3, 3), activation='relu', name = "conv3_" + str(feature)))
    # model.add(layers.BatchNormalization(name = "batch4_" + str(feature)))
    # model.add(layers.MaxPooling2D((2, 2), name = "max2_" + str(feature)))

    # model.add(layers.Conv2D(int(image_size/4), (3, 3), activation='relu', name = "conv4_" + str(feature)))
    # model.add(layers.BatchNormalization(name = "batch5_" + str(feature)))
    model.add(layers.Conv2D(int(image_size/8), (3, 3), activation='relu', name = "conv5_" + str(feature)))
    model.add(layers.BatchNormalization(name = "batch6_" + str(feature)))
    model.add(layers.MaxPooling2D((2, 2), name = "max3_" + str(feature)))

    model.add(layers.Conv2D(int(image_size/16), (3, 3), activation='relu', name = "conv6_" + str(feature)))
    model.add(layers.BatchNormalization(name = "batch7_" + str(feature)))
    # model.add(layers.Conv2D(int(image_size/8), (3, 3), activation='relu', name = "conv7_" + str(feature)))
    # model.add(layers.BatchNormalization(name = "batch8_" + str(feature)))
    # model.add(layers.MaxPooling2D((2, 2), name = "max4_" + str(feature)))

    # model.add(layers.Conv2D(int(image_size/16), (3, 3), activation='relu', name = "conv8_" + str(feature)))
    # model.add(layers.BatchNormalization(name = "batch9_" + str(feature)))
    model.add(layers.Conv2D(int(image_size/32), (3, 3), activation='relu', name = "conv9_" + str(feature)))
    model.add(layers.BatchNormalization(name = "batch10_" + str(feature)))
    model.add(layers.AveragePooling2D((2, 2), name = "avg1_" + str(feature)))

    model.add(layers.Flatten(name = "flatten_" + str(feature)))
    model.add(layers.Dense(48, activation='relu', name = "dense1_" + str(feature)))
    model.add(layers.Dropout(0.3, name = "dropout1_" + str(feature)))
    
    model.add(layers.Dense(16, activation='relu', name = "dense2_" + str(feature)))
    model.add(layers.Dropout(0.5, name = "dropout2_" + str(feature)))

    model.add(layers.Dense(1, activation='sigmoid', name = "dense3_" + str(feature)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.FalsePositives(), 
                    tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()])

    return model


def load_data_eyes(image_folder_sick, image_folder_healthy, image_size):
    images_left, labels_left = load_shuffled_data(image_folder_sick, image_folder_healthy, image_size, "left")
    images_right, labels_right = load_shuffled_data(image_folder_sick, image_folder_healthy, image_size, "right")

    images = np.concatenate((images_left, images_right), axis = 0)
    labels = np.concatenate((labels_left, labels_right), axis = 0)

    permutation = np.random.permutation(len(images))

    return images[permutation], labels[permutation]
    

def save_history(save_path, history, feature):
    with open(save_path + str(feature) + "/history.pickle" , 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    
    image_folder_sick = 'data/parsed/sick'
    image_folder_healthy = 'data/parsed/healthy'
    image_folder_all_sick = 'data/parsed/all_sick'
    image_folder_all_healthy = 'data/parsed/all_healthy'
    image_folder_altered = 'data/parsed/altered'
    image_folder_altered_1 = 'data/parsed/altered_1'
    image_folder_cfd = 'data/parsed/cfd'
    save_path = 'categorization/model_saves/'
    image_size = 128
    face_features = ["mouth", "face", "skin", "eyes"]
    
    # model = make_model(image_size, "mouth")
    # model.summary()

    for feature in face_features:
        
        print("[INFO] Training %s" %(feature))
        
        if feature == "eyes":
            test_images, test_labels = load_data_eyes(image_folder_sick, image_folder_healthy, image_size)
            train_images, train_labels = load_data_eyes(image_folder_altered_1, image_folder_cfd, image_size)

        else:
            test_images, test_labels = load_shuffled_data(image_folder_sick, image_folder_healthy, image_size, feature)
            train_images, train_labels = load_shuffled_data(image_folder_altered_1, image_folder_cfd, image_size, feature)

        model = make_model(image_size, feature)
        model.summary()

        history = model.fit(train_images, train_labels, epochs=10, batch_size = 32, validation_data=(test_images, test_labels))
        
        model.save(save_path + str(feature) + "/save.h5")
        save_history(save_path, history, feature)




