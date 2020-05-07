import os
import sys

sys.path.append(os.getcwd())

from categorization.cnn import *
from categorization.stacking_model import *

if __name__ == "__main__":
    
    image_folder_sick = 'data/parsed/sick'
    image_folder_healthy = 'data/parsed/healthy'
    image_folder_all_sick = 'data/parsed/all_sick'
    image_folder_all_healthy = 'data/parsed/all_healthy'
    image_folder_altered = 'data/parsed/altered'
    image_folder_altered_1 = 'data/parsed/altered_1'
    image_folder_cfd = 'data/parsed/cfd'
    save_path = 'categorization/model_saves/'
    image_size = 128
    face_features = ["mouth", "face", "skin", "eyes"]
    
    for feature in face_features:
        
        print("[INFO] Training %s" %(feature))
        
        if feature == "eyes":
            test_images, test_labels = load_data_eyes(image_folder_sick, image_folder_healthy, image_size)
            train_images, train_labels = load_data_eyes(image_folder_altered_1, image_folder_cfd, image_size)

        else:
            test_images, test_labels = load_shuffled_data(image_folder_sick, image_folder_healthy, image_size, feature)
            train_images, train_labels = load_shuffled_data(image_folder_altered_1, image_folder_cfd, image_size, feature)

        model = make_model(image_size, feature)
        # model.summary()

        history = model.fit(train_images, train_labels, epochs=10, batch_size = 32, validation_data=(test_images, test_labels))
        
        model.save(save_path + str(feature) + "/save.h5")
        save_history(save_path, history, feature)
        
    save_path = 'categorization/model_saves/'
    image_folder_sick = 'data/parsed/sick'
    image_folder_healthy = 'data/parsed/healthy'
    face_features = ["mouth", "face", "skin", "eyes"]
    image_size = 128
        
    all_models = load_all_models(save_path, face_features)

    train_images, train_labels, test_images, test_labels = make_training_sets(
        face_features, image_folder_sick, image_folder_healthy, image_size)
    
    print("Finished loading sets...")

    stacked = define_stacked_model(all_models, face_features)
    
    print("Starting training...")

    history = stacked.fit(
        train_images, train_labels, epochs=10,
        validation_data=(test_images, test_labels))
    save_history(save_path, history, "stacked")
    stacked.save(save_path + "stacked/save.h5")