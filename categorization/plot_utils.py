import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
from pandas import DataFrame

def print_roc_curve(tprs, auc_sum, feature, folds, base_fpr=np.linspace(0, 1, 101)):
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
    plt.title("ROC Curve for {} model (AUC = {:.3f})".format(str(feature).capitalize(), auc_sum / folds))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig("data/plots/roc_{}.png".format(feature))


def print_confusion_matrix(pred, true, feature, num_folds):
    matrix = np.zeros((2, 2))
    for i in range(num_folds):
        for j in range(len(true)):
            if pred[i*10+j] == 1 and true[j] == 1:
                matrix[0][1] += 1
            if pred[i*10+j] == 1 and true[j] == 0:
                matrix[1][1] += 1
            if pred[i*10+j] == 0 and true[j] == 1:
                matrix[0][0] += 1
            if pred[i*10+j] == 0 and true[j] == 0:
                matrix[1][0] += 1
    df_cm = DataFrame(matrix, index=["Positives", "Negative"], columns=[
                         "Negative", "Positives"])
    plt.figure()
    ax = plt.axes()
    heatmap(df_cm, annot=True, ax=ax, fmt='g', vmin = 0.0, vmax = 380.0)
    ax.set_title('Confusion Matrix ' + str(feature).capitalize())
    ax.set_ylabel("Actual Values")
    ax.set_xlabel("Predicted Values")
    plt.savefig("data/plots/confusion_matrix_" + str(feature) + ".png")
