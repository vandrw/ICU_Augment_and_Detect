import os
import sys

sys.path.append(os.getcwd())

from categorization.cnn import *
from categorization.stacking_model import *

if __name__ == "__main__":
    
    folder_sick_cnn = 'data/parsed/sick_1'
    folder_healthy_cnn = 'data/parsed/healthy_1'
    folder_sick_stacked = 'data/parsed/sick_2'
    folder_healthy_stacked = 'data/parsed/healthy_2'
    save_path = 'categorization/model_saves/'
    image_size = 128
    face_features = ["mouth", "face", "skin", "eyes"]
    

    for feature in face_features:
        
        print("[INFO] Training %s" %(feature))

        
        if feature == "eyes":
            all_images, all_labels = load_data_eyes(folder_sick_cnn, folder_healthy_cnn, image_size)

        else:
            all_images, all_labels = load_shuffled_data(folder_sick_cnn, folder_healthy_cnn, image_size, feature)

        train = int(len(all_images)*90/100)

        model = make_model(image_size, feature)

        history = model.fit(all_images[:train], all_labels[:train], epochs=10, batch_size = 32, validation_data=(all_images[train:], all_labels[train:]))
        
        model.save(save_path + str(feature) + "/save.h5")
        save_history(save_path, history, feature)
        
   
        
    all_models = load_all_models(save_path, face_features)

    train_images, train_labels, test_images, test_labels = make_training_sets(
        face_features, folder_sick_stacked, folder_healthy_stacked, image_size)
    
    print("Finished loading sets...")

    stacked = define_stacked_model(all_models, face_features)
    
    print("Starting training...")

    history = stacked.fit(
        train_images, train_labels, epochs=10,
        validation_data=(test_images, test_labels))
    save_history(save_path, history, "stacked")
    stacked.save(save_path + "stacked/save.h5")