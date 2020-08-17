
#%%
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
import random
import pydot 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from numpy import interp
from sklearn.model_selection import train_test_split
import copy

sys.path.append(os.getcwd())

from categorization.cnn import make_model, load_data, save_history, plot_roc


def load_all_models(save_path, features):
	all_models = list()
	for feature in features:
		# filename = save_path + str(feature) + '/save.h5'
		filename = save_path + str(feature) + '/model.h5'
		model = tf.keras.models.load_model(filename)
		all_models.append(model)
		print('loaded model of ' + str(feature))
	return all_models

def define_stacked_model(neural_nets, features):
    for model in neural_nets:
        for layer in model.layers:
            layer.trainable = False

    ensemble_visible = [model.input for model in neural_nets]
    ensemble_outputs = [model.layers[18].output for model in neural_nets]

    merge = tf.keras.layers.concatenate(ensemble_outputs)
    hidden = tf.keras.layers.Dense(32, activation='relu')(merge)
    hidden2 = tf.keras.layers.Dense(16, activation='relu')(hidden)
    hidden3 = tf.keras.layers.Dense(4, activation='relu')(hidden2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
    model = tf.keras.Model(inputs=ensemble_visible, outputs=output)

    # plot_model(model, show_shapes=True, to_file='data/plots/model_graph.png')
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    return model


# def make_training_sets(face_features, image_folder_sick, image_folder_healthy, image_size):
#     train_sets_images = []
#     train_sets_labels = []
#     for feature in face_features:
#         print("[INFO] loading %s" %(feature))
#         if feature == "eyes":
#             train_images, train_labels = load_data_eyes(image_folder_sick, image_folder_healthy, image_size)
#             size = int(len(train_images)/2)
#             train_images = train_images[:size]
#             train_sets_labels.append(train_labels[:size])
#         else:
#             train_images, train_labels = load_data(image_folder_sick, image_folder_healthy, image_size, feature)
#         train_sets_images.append(train_images)
    
#     return train_sets_images, train_sets_labels

def make_training_sets(face_features, image_folder_sick, image_folder_healthy, image_folder_val_sick, image_folder_val_healthy,image_size):

    train_images_mouth, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "mouth")
    train_images_nose, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "nose")
    train_images_skin, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "skin")
    train_images_right_eye, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "_right")

    test_images_mouth, test_labels = load_data(image_folder_val_sick, image_folder_val_healthy, image_size, "mouth")
    test_images_nose, test_labels = load_data(image_folder_val_sick, image_folder_val_healthy, image_size, "nose")
    test_images_skin, test_labels = load_data(image_folder_val_sick, image_folder_val_healthy, image_size, "skin")
    test_images_right_eye, test_labels = load_data(image_folder_val_sick, image_folder_val_healthy, image_size, "_right")

    all_test_labels = copy.deepcopy(test_labels)

    cross_val_images_mouth, test_images_mouth, cross_val_labels, test_labels = train_test_split(test_images_mouth, all_test_labels, test_size=0.5)
    cross_val_images_nose, test_images_nose, cross_val_labels, test_labels = train_test_split(test_images_nose, all_test_labels, test_size=0.5)
    cross_val_images_skin, test_images_skin, cross_val_labels, test_labels = train_test_split(test_images_skin, all_test_labels, test_size=0.5)
    cross_val_images_right_eye, test_images_right_eye, cross_val_labels, test_labels = train_test_split(test_images_right_eye, all_test_labels, test_size=0.5)

    perm1 = np.random.permutation(len(test_images_mouth))
    cross_val_images = [cross_val_images_mouth[perm1], cross_val_images_nose[perm1], cross_val_images_skin[perm1], cross_val_images_right_eye[perm1]]
    cross_val_labels = cross_val_labels[perm1]

    test_images = [test_images_mouth, test_images_nose, test_images_skin, test_images_right_eye]

    perm2 = np.random.permutation(len(train_images_mouth))
    train_images = [train_images_mouth[perm2], train_images_nose[perm2], train_images_skin[perm2], train_images_right_eye[perm2]]
    train_labels = train_labels[perm2]

    return train_images, train_labels, cross_val_images, cross_val_labels, test_images, test_labels


#%%
if __name__ == "__main__":

    image_folder_sick = 'data/parsed/brightened/sick'
    image_folder_healthy = 'data/parsed/brightened/healthy'
    image_folder_val_sick = 'data/parsed/validation_sick'
    image_folder_val_healthy = 'data/parsed/validation_healthy'
    save_path = 'categorization/model_saves/'
    face_features = ["mouth", "nose", "skin", "eyes"]
    image_size = 128

    all_models = load_all_models(save_path, face_features)

    train_images, train_labels, cross_val_images, cross_val_labels, test_images, test_labels = make_training_sets(
        face_features, image_folder_sick, image_folder_healthy, image_folder_val_sick, image_folder_val_healthy, image_size)

    auc_sum = 0
    cross_val_runs = 10

    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    
    for i in range(cross_val_runs):
        stacked = define_stacked_model(all_models, face_features)
        
        monitor = "val_accuracy"
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = monitor, mode = 'max', patience=10, verbose = 1)
        model_check = tf.keras.callbacks.ModelCheckpoint(save_path + 'stacked/model.h5', monitor=monitor, mode='max', verbose=1, save_best_only=True)
        
        print("Starting training...")

        history = stacked.fit(
            train_images, train_labels, epochs=50, callbacks=[model_check, early_stopping],
            validation_data=(cross_val_images, cross_val_labels), verbose = 1)

        
        save_history(save_path, history, "stacked", 4)

        stacked = tf.keras.models.load_model(save_path + 'stacked/model.h5')
    
    
        #  load best model as stacked to plot AUC


        pred = stacked.predict(test_images)
        
        fpr, tpr, _ = roc_curve(test_labels, pred)
        auc_sum += auc(fpr, tpr)

        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std


    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title("ROC Curve averaged over {} runs (Avg. AUC = {:.3f}".format(cross_val_runs, auc_sum / cross_val_runs))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig("data/plots/roc_stacked.png")