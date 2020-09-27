import os
import cv2
import numpy as np
import pickle

def load_data(folder_sick, folder_healthy, image_size, ftype):
    files_healthy = os.listdir(folder_healthy)
    files_sick = os.listdir(folder_sick)
    data = []
    labels = []
    for filename in files_healthy:
        sick = np.array([0])
        full_path = folder_healthy + "/" + str(filename)
        if ftype in filename and os.path.isfile(full_path) and "n2" not in filename:
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(
                image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype=np.int32))
            labels.append(np.asarray(sick, dtype=np.int32))
    for filename in files_sick:
        sick = np.array([1])
        full_path = folder_sick + "/" + str(filename)
        if ftype in filename and os.path.isfile(full_path):
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(
                image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype=np.int32))
            labels.append(np.asarray(sick, dtype=np.int32))
    return np.asarray(data, dtype=np.float64) / 255, np.asarray(labels, dtype=np.int32)


def make_stacked_sets(image_folder_sick, image_folder_healthy, image_size):

    train_images_mouth, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "mouth")
    train_images_nose, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "nose")
    train_images_skin, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "skin")
    train_images_right_eye, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "_right")

    perm = np.random.permutation(len(train_images_mouth))
    train_images = [train_images_mouth[perm], train_images_nose[perm],
                    train_images_skin[perm], train_images_right_eye[perm]]
    train_labels = train_labels[perm]

    return np.asarray(train_images), np.asarray(train_labels)


def load_shuffled_data(folder_sick, folder_healthy, image_size, ftype):
    data, labels = load_data(folder_sick, folder_healthy, image_size, ftype)
    permutation = np.random.permutation(len(data))
    return data[permutation], labels[permutation]

def save_history(save_path, history, feature, i):
    if i < 3:
        with open(save_path + str(feature) + "/history_" + str(i) + ".pickle", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    else:
        with open(save_path + str(feature) + "/history.pickle", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


def to_labels(predictions):
    pred = np.zeros((len(predictions), 1))
    for i in range(len(predictions)):
        if predictions[i] < 0.5:
            pred[i] = 0
        else:
            pred[i] = 1
    return pred
