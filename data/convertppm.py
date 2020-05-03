import os
import sys
import cv2

if len(sys.argv) == 1:
    print("Specify the location of the folder in which you want to convert the images...")
    print("Example: python convertppm.py [path_to_folder]")
    sys.exit()
    
if len(sys.argv) > 2:
    print("More than one directory was specified. Aborting.")
    sys.exit()
    
folder_path = sys.argv[1]
    
if not (os.path.exists(folder_path)):
    print('"', folder_path, '" does not exist!')
    sys.exit()

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        extension = file_name.split(".")[-1]
        if extension.lower() == "ppm":
            print("[INFO] Converting ", file_name)
            full_img_path = root + os.sep + file_name
            img = cv2.imread(full_img_path)
            new_path = root + os.sep + file_name.split(".")[0] + ".png"
            cv2.imwrite(new_path, img)