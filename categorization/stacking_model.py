
#%%
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
import random
import pydot

sys.path.append(os.getcwd())
from categorization.cnn import make_model, load_data, save_history


def load_all_models(save_path, features):
    all_models = list()
    for feature in features:
        filename = save_path + str(feature) + '/save.h5'
        model = tf.keras.models.load_model(filename)
        all_models.append(model)
        print('loaded model of ' + str(feature))
    return all_models


def define_stacked_model(neural_nets, features):
    for model in neural_nets:
        for layer in model.layers:
            layer.trainable = False

    ensemble_visible = [model.input for model in neural_nets]
    ensemble_outputs = [model.layers[27].output for model in neural_nets]

    merge = tf.keras.layers.concatenate(ensemble_outputs)
    hidden = tf.keras.layers.Dense(128, activation='relu')(merge)
    hidden2 = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden3 = tf.keras.layers.Dense(4, activation='relu')(hidden2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
    model = tf.keras.Model(inputs=ensemble_visible, outputs=output)

    plot_model(model, show_shapes=True, to_file='data/plots/model_graph.png')
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    return model


def import_data(path):
    img_dict = {}

    for root, dirs, files in os.walk(path):

        for file_name in files:
            if file_name == ".gitkeep":
                continue

            split = file_name.split("_")
            if split[1].split(".")[0] == "left":
                continue

            name = split[0].lower()
            full_path = path = root + os.sep + file_name

            img = np.asarray(cv2.imread(full_path))
            if (name in img_dict):
                img_dict[name].append(img)
            else:
                img_dict[name] = [img]

    return img_dict

def make_training_sets(face_features, image_folder_sick, image_folder_healthy, image_size):

    train_images_mouth, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "mouth")
    train_images_face, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "face")
    train_images_skin, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "skin")
    train_images_right_eye, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "right")

    permutation = np.random.permutation(len(train_labels))
    train_images_mouth = train_images_mouth[permutation]
    train_images_face = train_images_face[permutation]
    train_images_skin = train_images_skin[permutation]
    train_images_right_eye = train_images_right_eye[permutation]
    train_labels = train_labels[permutation]

    size = len(train_labels)-15

    train_images = [train_images_mouth[:size], train_images_face[:size],
                    train_images_skin[:size], train_images_right_eye[:size]]

    test_images = [train_images_mouth[size:], train_images_face[size:],
                   train_images_skin[size:], train_images_right_eye[size:]]

    return train_images, train_labels[:size], test_images, train_labels[size:]


#%%
if __name__ == "__main__":
    save_path = 'categorization/model_saves/'
    image_folder_sick = 'data/parsed/sick'
    image_folder_healthy = 'data/parsed/healthy'
    face_features = ["mouth", "face", "skin", "eyes"]
    image_size = 217
    
        
    # img_dict = import_data(image_folder_sick)

    # print(len(img_dict["s1s-m"]))
    
    # for x, y in img_dict.items():
    #     img = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

    all_models = load_all_models(save_path, face_features)

    train_images, train_labels, test_images, test_labels = make_training_sets(
        face_features, image_folder_sick, image_folder_healthy, image_size)
    
    print("Finished loading sets...")

    stacked = define_stacked_model(all_models, face_features)
    
    print("Starting training...")

    history = stacked.fit(
        train_images, train_labels, epochs=100, verbose=0,
        validation_data=(test_images, test_labels))
    save_history(save_path, history, "stacked")
    stacked.save(save_path + "stacked/save.h5")
