'''
- flip original mouth/face/skin
- use one eye + not flipped images for training/testing stacked model
- for testing use the individual testing sets for CNNs + stacked testing set
- early stopping for CNNs
- keep best model for graphs
'''

import os
import sys
import random

sys.path.append(os.getcwd())

from categorization.cnn import *
from categorization.stacking_model import *

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

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = monitor, mode = 'max', patience=10, verbose = 1)
        model_check = tf.keras.callbacks.ModelCheckpoint(save_path + str(feature)+ '/model.h5', monitor=monitor, mode='max', verbose=1, save_best_only=True)

        history = model.fit(train_images, train_labels, epochs=50,
                            batch_size=8, callbacks = [early_stopping, model_check], validation_data=(test_images, test_labels))

        save_history(save_path, history, feature)

        saved_model = tf.keras.models.load_model(save_path + str(feature)+ '/model.h5')

    print("Loading the stacked model...")

    all_models = load_all_models(save_path, face_features)

    train_images, train_labels, test_images, test_labels = make_training_sets(face_features, image_folder_sick, image_folder_healthy, image_folder_val_sick, image_folder_val_healthy, image_size)

    stacked = define_stacked_model(all_models, face_features)
    
    monitor = "val_accuracy"
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = monitor, mode = 'max', patience=10, verbose = 1)
    model_check = tf.keras.callbacks.ModelCheckpoint(save_path + 'stacked/model.h5', monitor=monitor, mode='max', verbose=1, save_best_only=True)
    
    print("Starting training...")

    history = stacked.fit(
        train_images, train_labels, epochs=50, callbacks=[model_check, early_stopping],
        validation_data=(test_images, test_labels), verbose = 1)

    
    save_history(save_path, history, "stacked")

    stacked = tf.keras.models.load_model(save_path + 'stacked/model.h5')
    
    
    #  load best model as stacked to plot AUC


    pred = stacked.predict(test_images)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(test_labels, pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("data/plots/best_stacked_auc.png")
