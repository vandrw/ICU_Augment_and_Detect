import os
import sys
import random

sys.path.append(os.getcwd())

from categorization.cnn import *
from categorization.stacking_model import *

if __name__ == "__main__":

    sick_1 = 'data/parsed/sick_1'
    healthy_1 = 'data/parsed/healthy_1'
    sick_2 = 'data/parsed/sick_2'
    healthy_2 = 'data/parsed/healthy_2'
    save_path = 'categorization/model_saves/'
    face_features = ["mouth", "face", "skin", "eyes"]
    image_size = 128
    cross_validation = 11

    folder_sick_cnn = sick_1
    folder_healthy_cnn = healthy_1
    folder_sick_stacked = sick_2
    folder_healthy_stacked = healthy_2

    for i in range(1, cross_validation):

        # if random.uniform(0, 1) < 0.5:
        #     folder_sick_cnn = sick_1
        #     folder_healthy_cnn = healthy_1
        #     folder_sick_stacked = sick_2
        #     folder_healthy_stacked = healthy_2
        # else:
        #     folder_sick_cnn = sick_2
        #     folder_healthy_cnn = healthy_2
        #     folder_sick_stacked = sick_1
        #     folder_healthy_stacked = healthy_1

        for feature in face_features:

            print("[INFO] Training %s" % (feature))

            if feature == "eyes":
                all_images, all_labels = load_data_eyes(
                    folder_sick_cnn, folder_healthy_cnn, image_size)

            else:
                all_images, all_labels = load_shuffled_data(
                    folder_sick_cnn, folder_healthy_cnn, image_size, feature)

            train = int(len(all_images)*90/100)

            model = make_model(image_size, feature)

            history = model.fit(all_images[:train], all_labels[:train], epochs=10, batch_size=16, validation_data=(
                all_images[train:], all_labels[train:]))

            model.save(save_path + str(feature) + "/save_" + str(i) + ".h5")
            save_history(save_path, history, feature, i)

        train_images, train_labels, test_images, test_labels = make_training_sets(
            face_features, folder_sick_stacked, folder_healthy_stacked, image_size)

        print("Finished loading sets...")

        all_models = load_all_models(save_path, face_features, i)

        stacked = define_stacked_model(all_models, face_features)

        print("Starting training...")

        history = stacked.fit(
            train_images, train_labels, epochs=10,
            validation_data=(test_images, test_labels))
        save_history(save_path, history, "stacked", i)
        stacked.save(save_path + "/save_" + str(i) + ".h5")
