import os
import sys
import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

sys.path.append(os.getcwd())
from categorization.models import make_model
from categorization.plot_utils import *
from categorization.data_utils import *



if __name__ == "__main__":

    image_folder_val_sick = 'data/parsed/validation_sick'
    image_folder_val_healthy = 'data/parsed/validation_healthy'
    save_path = 'categorization/model_saves/'
    face_features = ["mouth", "nose", "skin", "eye"]
    
    base_fpr = np.linspace(0, 1, 101)
    image_size = 128
    folds = 10

    for feature in face_features:
        auc_sum = 0
        tprs = []

        print("[INFO] Making plots for %s" % (feature))

        val_images, val_labels = load_shuffled_data(
            image_folder_val_sick, image_folder_val_healthy, image_size, feature)

        for fold_no in range(1,folds+1):

            tf.keras.backend.clear_session()
            best_model_path = save_path + str(feature) + "/model_" + str(fold_no) + '.h5'
            saved_model = tf.keras.models.load_model(best_model_path)

            if fold_no == 1:
                predictions = to_labels(saved_model.predict(val_images))
            else:
                predictions = np.concatenate((predictions, to_labels(saved_model.predict(val_images))), axis=0)

            pred = (saved_model.predict(val_images))
            fpr, tpr, _ = roc_curve(val_labels, pred)
            auc_sum += auc(fpr, tpr)
            del saved_model

            plt.plot(fpr, tpr, 'b', alpha=0.15)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
        
        plt.figure()
        print_roc_curve(tprs, auc_sum, feature, folds)
        print_confusion_matrix(predictions, val_labels, feature, folds)
