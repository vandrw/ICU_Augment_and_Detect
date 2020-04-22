"""
Script that calculates:
 * a circle enclosing each eye, along with its corresponding eyebrow,
 * a circle enclosing the lips,
 * the skin pixels, without eyebrows, eyes or mouth.
 
The right-eye regions are flipped horizontally, as there is a single
generator for the entire eye.

Building upon Shuvrajit9904's work:
https://github.com/Shuvrajit9904/PairedCycleGAN-tf/blob/master/parse_face.py

!!! BACK-UP FILE !!!
"""
#%%
from imutils import face_utils as futil
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt

mouth_ids = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])

right_eye_region = np.array([17, 18, 19, 20, 21, 39, 40, 41, 36])
left_eye_region = np.array([22, 23, 24, 25, 26, 45, 46, 47, 42])

jaw_ids = np.arange(0, 17)

FACIAL_LANDMARKS_IDXS = OrderedDict ([
	("mouth", mouth_ids),
    ("right_eye_region", right_eye_region),
    ("left_eye_region", left_eye_region),
    ("jaw", jaw_ids)
])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('augment/parsing/shape_predictor_68_face_landmarks.dat')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getDominantColor(img):
    data = np.reshape(img, (-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)
    
    return tuple([int(x) for x in centers[0].astype(np.int32)])

def readAndResize(image_path, target_size=512):
    img = cv2.imread(image_path)
    if (img.size == 0):
        print("The image could not be loaded!")
        return
    
    if (img.shape[1] < img.shape[0]):
        min_dim = img.shape[1]
    else:
        min_dim = img.shape[0]
    
    if (min_dim > target_size):
        scale = target_size / min_dim
        
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
         
        img = cv2.resize(img, 
                         new_size,
                         interpolation=cv2.INTER_AREA)
        
        centerY = int(img.shape[0] / 2)
        centerX = int(img.shape[1] / 2)
        
        if (centerX > centerY):
            rightX = int(centerX + (target_size / 2))
            leftX =  int(centerX - (target_size / 2))
            img = img[:, leftX:rightX]
        else:
            bottomY = int(centerY + (target_size / 2))
            topY =    int(centerY - (target_size / 2))
            img = img[bottomY:topY, :]
        
    return img

img = readAndResize("augment/style_sick1.png")
img2 = readAndResize("augment/style_normal.jpg")
img3 = readAndResize("augment/as3.jpg")
img4 = readAndResize("augment/a4.jpg")

# %%

def extractFeatures(img, detector, predictor, dominant_color):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rectangles = detector(gray_img, 1)
    
    shape = predictor(gray_img, rectangles[0])
    shape = futil.shape_to_np(shape)

    for (name, id_arr) in FACIAL_LANDMARKS_IDXS.items():

        clone = img.copy()
        
        if (name != "jaw"):
            (x,y), radius = cv2.minEnclosingCircle(np.array([shape[id_arr]]))  
            center = (int(x),int(y))  
            radius = int(radius) + 20   
            
            mask = np.zeros(clone.shape, dtype=np.uint8)  
            mask = cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)
            
            fig, (ax1, ax2) = plt.subplots(1,2)
            
            ax1.imshow(mask)
            
            result_array = (clone & mask)
            y_min = max(0, center[1] - radius)
            x_min = max(0, center[0] - radius)
            result_array = result_array[y_min:center[1] + radius,
                                x_min:center[0] + radius, :]
            
            result_array[np.where((result_array==[0,0,0]).all(axis=2))] = dominant_color

            if name == 'left_eye_region':
                ax2.imshow(cv2.cvtColor(cv2.flip(result_array, 1), cv2.COLOR_BGR2RGB))

            else:
                ax2.imshow(cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB))
            
            plt.figure()
        else:
            mask = np.zeros(clone.shape, dtype=np.uint8)
            cv2.drawContours(mask, [shape[jaw_ids]], 0, (255,255,255), -1, 8)

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(mask)

            result_array = clone & mask
            result_array[np.where((result_array==[0,0,0]).all(axis=2))] = dominant_color
            ax2.imshow(cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB))
            plt.figure()
            
    return shape
            

# ab = extractFeatures(img2, detector, predictor, [124,124,124])      
# %%
def extractFace(img, faceCascade, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(200, 200)
    )
    
    print("[INFO] Found {0} Faces.".format(len(faces)))
    
    (x, y, w, h) = faces[0]
    
    copy = img.copy()
    
    face = copy[y:y + h, x:x + w]
    # cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    dominant_color = getDominantColor(face)
    
    shapeFeatures = extractFeatures(face, detector, predictor, dominant_color)
    
    plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
    plt.figure()
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    blur = cv2.blur(face,(10,10))
    ax2.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.figure()
    
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    face_copy = face.copy()
    
    kernel = np.ones((8,8),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    
    face_copy[dilation == 255] = dominant_color
    
    
    plt.imshow(cv2.cvtColor(face_copy, cv2.COLOR_BGR2RGB))
    plt.figure()
    
    _, thresh = cv2.threshold(cv2.cvtColor(face_copy, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((10,10),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    
    face_copy[dilation == 0] = dominant_color
    # plt.imshow(thresh)
    
    for (name, id_arr) in FACIAL_LANDMARKS_IDXS.items():
        if (name != "jaw"):
            cv2.fillPoly(face_copy, pts=[shapeFeatures[id_arr]], color=dominant_color)
        else:
            
            img_shape = face_copy.shape
            
            jaw_pixels = shapeFeatures[jaw_ids]
            outside_pixels = np.array([[0, img_shape[1]], [0,0], [15, 0]])
            
            outside_pixels = np.append(outside_pixels, jaw_pixels, axis=0)
            
            outside_pixels = np.append(outside_pixels, [[img_shape[0] - 15, 0], [img_shape[0], 0], [img_shape[0], img_shape[1]]], axis=0)
                          
            cv2.fillPoly(face_copy, pts=[outside_pixels], color=dominant_color)

    plt.imshow(cv2.cvtColor(face_copy, cv2.COLOR_BGR2RGB))
    plt.figure()
    
    # print(outside_pixels)
    

extractFace(img3, faceCascade, detector, predictor)