import os
import sys
from numpy import interp
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.getcwd())
from categorization.models import *
from categorization.plot_utils import *
from categorization.data_utils import *

if __name__ == "__main__":

    image_folder_testing_sick = 'data/parsed/brightened/sick'
    image_folder_testing_healthy = 'data/parsed/brightened/healthy'
    image_folder_training_sick = 'data/parsed/sick'
    image_folder_training_healthy = 'data/parsed/healthy'
    image_folder_val_sick = 'data/parsed/validation_sick'
    image_folder_val_healthy = 'data/parsed/validation_healthy'
    save_path = 'categorization/model_saves/'
    face_features = ["mouth", "nose", "skin", "eyes"]
    base_fpr = np.linspace(0, 1, 101)
    image_size = 128
    folds = 5
    auc_sum = 0
    tprs = []

    kfold = KFold(n_splits=folds, shuffle=True, random_state=1)

    plt.figure()

    #test_faces, _ = load_data(
    #    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "face")

    test_images, test_labels = make_stacked_sets(image_folder_testing_sick, image_folder_testing_healthy, image_size)
    train_images, train_labels = make_stacked_sets(image_folder_training_sick, image_folder_training_healthy, image_size)
    val_images, val_labels = make_stacked_sets(image_folder_val_sick, image_folder_val_healthy, image_size)

    print("Creating empty models...")
    for feature in face_features:
        print(feature + "...")
        model = make_model(image_size, feature)
        model.save(save_path + os.sep + feature + os.sep + "model.h5")

    fold_no = 1
    # for train, test in kfold.split(images[0], labels):

    print("Loading the stacked model...")

    all_models = load_all_models(save_path, face_features)

    stacked = define_stacked_model(all_models, face_features)

    early_stopping = EarlyStopping(
        monitor="val_accuracy", mode='max', patience=10, verbose=1)
    model_check = ModelCheckpoint(save_path + 'stacked/model_' + str(
        fold_no) + '.h5', monitor="val_accuracy", mode='max', verbose=1, save_best_only=True)

        print("Starting training...")

    # history = stacked.fit(
    #     x=[images[0, train], images[1, train], images[2, train], images[3, train]],
    #     y=labels[train], epochs=50, batch_size=1, callbacks=[early_stopping, model_check],
    #     validation_data=([images[0, test], images[1, test], images[2, test], images[3, test]], labels[test]))

    history = stacked.fit(x = train_images, y = train_labels, validation_data = (test_images, test_labels))
    save_history(save_path, history, "stacked", fold_no)

    print("Loading model and making predictions...")
    #  load best model as stacked to plot predictions
    stacked = tf.keras.models.load_model(
        save_path + 'stacked/model_' + str(fold_no) + '.h5')

    # pred = stacked.predict(
    #     [test_images[0], test_images[1], test_images[2], test_images[3]])
    
    pred = stacked.predict(val_images)
    if fold_no == 1:
        predictions = to_labels(pred)
    else:
        predictions = np.concatenate(
            (predictions, to_labels(pred)), axis=0)

    # fold_no += 1

    fpr, tpr, _ = roc_curve(val_labels, pred)
    auc_sum += auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

    print_roc_curve(tprs, auc_sum, "stacked", fold_no)
    print_confusion_matrix(predictions, val_labels, "stacked", fold_no)
