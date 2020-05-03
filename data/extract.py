import os
import sys
import shutil

if len(sys.argv) == 1:
    print("Specify the location of the folder you want to extract the images from...")
    print("Example: python extract.py [path_to_folder] [destination_folder]")
    sys.exit()
    
if len(sys.argv) == 2:
    print("The script will extract all the images in:")
    print(os.path.abspath("."))
    ans = input("Do you wish to continue?[y/n] ")
    if not (ans == "y"):
        sys.exit()

if len(sys.argv) > 3:
    print("More than one directory was specified. Aborting.")
    sys.exit()
    
if len(sys.argv) == 3:
    target_path = sys.argv[2]
    if not (os.path.exists(target_path)):
        print('"', target_path, '" does not exist!')
        sys.exit()
    
folder_path = sys.argv[1]
image_extensions = ["jpg", "png", "ppm"]
    
if not (os.path.exists(folder_path)):
    print('"', folder_path, '" does not exist!')
    sys.exit()

# If the file names are the same, set this to true to change
# the name of the files after they are copied.
sameName = True
usePreviousFolder = True

if not usePreviousFolder:
    i = 0

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(folder_path):
    # print("Extracting from ", root.split(os.sep)[-1])
    for file_name in files:
        extension = file_name.split(".")[-1]
        if extension.lower() in image_extensions:
            print("[INFO] Moving ", file_name)
            full_img_path = root + os.sep + file_name
            shutil.copy(full_img_path, target_path)
            if sameName:
                target_img_path = target_path + os.sep + file_name
                
                if usePreviousFolder:
                    new_img = target_path + os.sep + root.split(os.sep)[-1] + file_name
                else:
                    new_img = target_path + os.sep + str(i) + file_name
                    i += 1
                    
                shutil.move(target_img_path, new_img)
        # if file_name.split(".")[-1] == "bz2":
        #     full_img_path = root + os.sep + file_name
        #     os.system("data/colorferet/colorferet/dvd1/source/bzip2-1.0.2/bzip2 -d " + full_img_path)