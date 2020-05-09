# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %%
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
import pickle
import csv
import seaborn as sn
import pandas as pd
import re
import math
sys.path.append(os.getcwd())


def load_histories(save_path):
    history = '/history.pickle'
    models = ["mouth", "face", "skin", "eyes", "stacked"]
    all_histories = {}
    for model in models:
        all_histories[model] = load_average(save_path, model)
    return all_histories


# %%
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def rename_keys(dictionary):
    new_dictionary = {}
    for key in dictionary:
        if hasNumbers(key):
            new_key = "_".join(key.split("_")[:-1])
            new_dictionary[new_key] = np.asarray(dictionary[key])
        else:
            new_dictionary[key] = np.asarray(dictionary[key])
    return new_dictionary

def load_average(save_path, model):
    path = save_path + str(model)
    files = os.listdir(path)
    i = 0
    sum_histories = {}
    for f in files:
        if "pickle" in f:
            hist_path = path + "/history_" + str(i+1) + ".pickle"
            if not os.path.isfile(hist_path):
                hist_path = path + "/history.pickle"
            hist_file = open(hist_path, "rb")
            history = pickle.load(hist_file)
            history = rename_keys(history)
            if i == 0:
                sum_histories = history
            else:
                for key in history:
                    sum_histories[key] += history[key]
            i += 1
    for key in sum_histories:
        sum_histories[key]= sum_histories[key]/i

    return sum_histories

# %%
def print_raw(all_histories):
    with open("data/exact_values.csv", "w") as data_file:
        writer = csv.writer(data_file, delimiter=',')
        header = ['Model', 'Training Accuracy', 'Validation Accuracy', "Training Precision", "Validation Precision", "Training Recall", "Validation Recall", "Training F1", "Validation F1", "Training MCC", "Validation MCC"]
        writer.writerow(header)
        for model in all_histories:
            final = len(all_histories[model]["accuracy"]) - 1
            acc = all_histories[model]["accuracy"][final]
            val_acc = all_histories[model]["val_accuracy"][final]
            val_tp = all_histories[model]["val_true_positives"][final]
            val_tn = all_histories[model]["val_true_negatives"][final]
            val_fp = all_histories[model]["val_false_positives"][final]
            val_fn = all_histories[model]["val_false_negatives"][final]
            tp = all_histories[model]["true_positives"][final]
            tn = all_histories[model]["true_negatives"][final]
            fp = all_histories[model]["false_positives"][final]
            fn = all_histories[model]["false_negatives"][final]
            precision =  tp / (tp+fp)
            val_precision =  val_tp / (val_tp+val_fp)
            recall =  tp / (tp+fn)
            val_recall =  val_tp / (val_tp+val_fn)
            f1 = 2 * precision * recall / (precision + recall)
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)
            den = math.sqrt((tp + fp) * (fn + tn) * (fp + tn) * (tp + fn))
            if den == 0:
                den = 
            mcc = (tp * tn - fp * fn)/
            val_mcc = (val_tp * val_tn - val_fp * val_fn)/math.sqrt((val_tp + val_fp) * (val_fn + val_tn) * (val_fp + val_tn) * (val_tp + val_fn))
            row = [str(model), acc, val_acc, precision, val_precision, recall, val_recall, f1, val_f1, mcc, val_mcc]
            writer.writerow(row)

def plot_confusion_matrix(all_histories):
    for model in all_histories:
        final = len(all_histories[model]["accuracy"]) - 1
        matrix = [[all_histories[model]["val_true_positives"][final]*10, all_histories[model]["val_false_positives"][final]*10],
                    [all_histories[model]["val_false_negatives"][final]*10, all_histories[model]["val_true_negatives"][final]*10]]
        df_cm = pd.DataFrame(matrix, index = ["Positives", "Negative"],
              columns = ["Positives", "Negative"])
        ax = plt.axes()
        sn.heatmap(df_cm, annot=True, ax=ax)
        ax.set_title('Confusion Matrix ' + str(model))
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        plt.savefig("data/plots/confusion_matrix_" + str(model) + ".png")
        plt.figure()
        # plt.show()

def plot_all_auc_acc(all_histories):

    fig = plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    for key in all_histories:
        plt.plot(all_histories[key]["accuracy"], label = str(key))
    plt.xlim((0,10))
    plt.xlabel('Training Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend(bbox_to_anchor=(0.2, 1.02, 1.8, .102), loc='lower left',
            ncol=5, mode="expand", borderaxespad=0., title = "Model")

    plt.subplot(2,2,2)
    for key in all_histories:
        plt.plot(all_histories[key]["auc"], label = str(key))
    plt.xlim((0,10))
    plt.xlabel('Training Epochs')
    plt.ylabel('Training AUC')

    plt.subplot(2,2,3)
    for key in all_histories:
        plt.plot(all_histories[key]["val_accuracy"], label = str(key))
    plt.xlim((0,10))
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Accuracy')

    plt.subplot(2,2,4)
    for key in all_histories:
        plt.plot(all_histories[key]["val_auc"], label = str(key))
    plt.xlim((0,10))
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation AUC')

    plt.savefig("data/plots/models_acc_auc.png")
    # plt.figure()
    # plt.show()

if __name__ == "__main__":

    save_path = 'categorization/model_saves/'

    all_histories = load_histories(save_path)
    plot_all_auc_acc(all_histories)
    plot_confusion_matrix(all_histories)
    print_raw(all_histories)


