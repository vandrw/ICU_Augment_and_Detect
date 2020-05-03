import os
import sys

separator = "-"
target_words = ["003F", "006F"]
position = 1

if len(sys.argv) == 1:
    print("Specify the location of the folder in which you want to delete the images...")
    print("Example: python delete.py [path_to_folder]")
    sys.exit()
    
if len(sys.argv) > 2:
    print("More than one argument was specified. Aborting.")
    sys.exit()
    
folder_path = sys.argv[1]
    
if not (os.path.exists(folder_path)):
    print('"', folder_path, '" does not exist!')
    sys.exit()

for root, dirs, files in os.walk(folder_path):   
    no_extension = files[0].split(".")[0]
    target = no_extension.split(separator)[position]
    print("The first file I found was: ", files[0])
    print('"', target, '" will be compared with the words specified.')
    print("If this does not match the words in ", target_words, ", the file will be deleted.")
    a = input("Do you wish to continue? [y/n] ")

    if a != "y":
        sys.exit()
    
    break

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        full_img_path = root + os.sep + file_name
        no_extension = file_name.split(".")[0]
        target = no_extension.split(separator)[position]
        if target not in target_words:
            print("[INFO] Deleting ", full_img_path)
            os.remove(full_img_path)
        