#%cd ..
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
from keras.layers.normalization import BatchNormalization
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
        if type in filename and os.path.isfile(full_path):
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

    events = list(zip(data, labels))
    random.shuffle(events)
    data, labels = zip(*events)
    # print(data)
    return np.asarray(data, dtype=np.int32), np.asarray(labels, dtype=np.int32)

def make_model(image_size):
    model = models.Sequential()

    model.add(layers.Conv2D(image_size, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(image_size, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(int(image_size/2), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(int(image_size/2), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(int(image_size/4), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(int(image_size/4), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(int(image_size/8), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(int(image_size/8), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(int(image_size/16), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(int(image_size/16), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(52, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='softmax'))

    return model

if __name__ == "__main__":
    image_folder_sick = 'data/parsed/sick'
    image_folder_healthy = 'data/parsed/healthy'
    image_folder_altered = 'data/parsed/altered'
    image_folder_cfd = 'data/parsed/cfd'
    checkpoint_path = 'categorization/model_saves/mouth.ckpt'
    image_size = 217
    key = "mouth"

    test_images_mouth, test_labels_mouth = load_data(image_folder_sick, image_folder_healthy, image_size, key)
    train_images_mouth, train_labels_mouth = load_data(image_folder_altered, image_folder_cfd, image_size, key)

    model = make_model(image_size)
    model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    history = model.fit(train_images_mouth, train_labels_mouth, epochs=100, 
                    validation_data=(test_images_mouth, test_labels_mouth), callbacks=[cp_callback])

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('data/plots/accuracy_mouth.png')

    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig('data/plots/learning_curve_mouth.png')



