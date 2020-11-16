import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
from pandas import DataFrame
import scipy.stats as st


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

def compute_confidence_int(values):
    return st.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=st.sem(values))

def print_confidence_intervals(pred, true, auc_values, feature, num_folds):
    results_file = "data/plots/confidence_intervals.csv"
    specificity = np.zeros(num_folds)
    sensitivity = np.zeros(num_folds)
    for i in range(num_folds):
        matrix = np.zeros((2, 2))
        for j in range(len(true)):
            if pred[i*10+j] == 1 and true[j] == 1:
                matrix[0][1] += 1
            if pred[i*10+j] == 1 and true[j] == 0:
                matrix[1][1] += 1
            if pred[i*10+j] == 0 and true[j] == 1:
                matrix[0][0] += 1
            if pred[i*10+j] == 0 and true[j] == 0:
                matrix[1][0] += 1
        sensitivity[i] = matrix[0][1]/(matrix[0][1] + matrix[0][0])
        specificity[i] = matrix[1][0]/(matrix[1][0] + matrix[1][1])
    f = open(results_file, 'a')
    f.write('Model for ' + str(feature) + '\n')
    ci = compute_confidence_int(sensitivity)
    f.write("Sensitivity = {:.3f} +/- {:.3f} 95% confidence interval = {} \n".format(np.mean(sensitivity), (ci[1]-ci[0])/2, ci))
    ci = compute_confidence_int(specificity)
    f.write("Specificity = {:.3f} +/- {:.3f} 95% confidence interval = {} \n".format(np.mean(specificity), (ci[1]-ci[0])/2, ci))
    ci = compute_confidence_int(auc_values)
    f.write("AUC = {:.3f} +/- {:.3f} 95% confidence interval = {} \n\n".format(np.mean(auc_values), (ci[1]-ci[0])/2, ci))
    f.close()

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
    heatmap(df_cm, annot=True, ax=ax, fmt='g', vmin = 0.0, vmax = 220.0)
    ax.set_title('Confusion Matrix ' + str(feature).capitalize())
    ax.set_ylabel("Actual Values")
    ax.set_xlabel("Predicted Values")
    plt.savefig("data/plots/confusion_matrix_" + str(feature) + ".png")


def plot_per_participant(per_participant, face_features):
    ind = np.arange(len(per_participant[0]))
    width = 0.5     

    bot = np.zeros(len(ind))
    plots = []

    plt.figure(figsize=(15, 7))
    for feature in face_features:
        p = plt.bar(ind, per_participant[face_features.index(feature)], width, bottom = bot)
        bot += per_participant[face_features.index(feature)]
        plots.append(p[0])
    plt.axvline(x=18.5, c='tab:gray', linestyle=':')
    plt.text(6, 4, 'Healthy', color = 'tab:green', fontsize= 20)
    plt.text(28, 4, 'Sick', color = 'tab:red', fontsize = 20)

    plt.ylabel('Accuracy per Model')
    plt.title('Accuracy per Participant per Model')

    # xticks = []
    # for i in range(len(ind)):
    #     xticks.append("P " + str(i+1))

    plt.xticks(ind, tuple([str(i+1) for i in ind]))
    plt.yticks(np.arange(0, len(face_features), 0.5))
    plt.legend(plots, [feature.capitalize() for feature in face_features])
    
    if len(face_features) == 1:
        plt.savefig("data/plots/acc_per_participant_stacked.png")
    else:
        plt.savefig("data/plots/acc_per_participant.png")