'''
- make each network more complex
V binary output
- cross validation
X increase size of images
- the other metrics
X try Marco's GAN
V extract nose and replace face with it
'''

'''
- bootstrapping: 1000x, choose different samples
- GAN on cfd
'''

import os
import sys
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from numpy import interp

sys.path.append(os.getcwd())

from categorization.cnn import *
from categorization.stacking_model import *

def get_accuracy(test_labels, prediction_labels):
    sum_acc = 0.0
    for i in range(len(test_labels)):
        if (test_labels[i] == (prediction_labels[i] >= 0.5)):
            sum_acc += 1
    
    return sum_acc / len(test_labels)

def print_confusion_matrix(pred, true, feature, num_folds):
    matrix = np.zeros((2,2))
    for i in range(num_folds):
        for j in range(len(true)):
            if pred[i*j] == 1 and true[j] == 1:
                matrix[0][0] += 1
            if pred[i*j] == 1 and true[j] == 0:
                matrix[0][1] += 1
            if pred[i*j] == 0 and true[j] == 1:
                matrix[1][0] += 1
            if pred[i*j] == 0 and true[j] == 0:
                matrix[1][1] += 1
    df_cm = pd.DataFrame(matrix, index = ["Positives", "Negative"], columns = ["Positives", "Negative"])
    plt.figure()
    ax = plt.axes()
    sn.heatmap(df_cm, annot=True, ax=ax, fmt='g')
    ax.set_title('Confusion Matrix ' + str(feature))
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    plt.savefig("data/plots/confusion_matrix_" + str(feature) + ".png")

if __name__ == "__main__":

    image_folder_sick = 'data/parsed/brightened/sick'
    image_folder_healthy = 'data/parsed/brightened/healthy'
    image_folder_val_sick = 'data/parsed/validation_sick'
    image_folder_val_healthy = 'data/parsed/validation_healthy'
    save_path = 'categorization/model_saves/'
    image_size = 128
    face_features = ["mouth", "nose", "skin", "eyes"]

    folds = 5
    kfold = KFold(n_splits=folds, shuffle=True, random_state=1)

    auc_sum = 0
    tprs = []

    fold_no = 1

    plt.figure()

    test_faces, _ = load_data(
        'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "face")

    images, labels, test_images, test_labels = make_training_sets(
        face_features, image_folder_sick, image_folder_healthy, image_folder_val_sick, image_folder_val_healthy, image_size)

    for train, test in kfold.split(images, labels):
        
        print("Creating empty models...")
        for feature in face_features:
            print(feature + "...")
            model = make_model(image_size, feature)
            model.save(save_path + os.sep + feature + os.sep + "model.h5")

        print("Loading the stacked model...")

        all_models = load_all_models(save_path, face_features)

        # print(cross_val_labels)

        # for i in range(3):
        stacked = define_stacked_model(all_models, face_features)
        
        monitor = "val_accuracy"
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = monitor, mode = 'max', patience=10, verbose = 1)
        model_check = tf.keras.callbacks.ModelCheckpoint(save_path + 'stacked/model.h5', monitor=monitor, mode='max', verbose=1, save_best_only=True)
        
        print("Starting training...")

        history = stacked.fit(
            train_images, train_labels, epochs=50, batch_size=2, callbacks=[model_check, early_stopping],
            validation_data=(cross_val_images, cross_val_labels), verbose = 1)

        
        # save_history(save_path, history, "stacked")

        print("Loading model and making predictions...")
        stacked = tf.keras.models.load_model(save_path + 'stacked/model.h5')
            
            
        #  load best model as stacked to plot predictions


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
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("ROC Curve for " + str(feature) + " (AUC = {:.3f})".format(auc_sum / folds))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig("data/plots/roc_stacked.png")

    print_confusion_matrix(predictions, val_labels, "stacked", folds)
