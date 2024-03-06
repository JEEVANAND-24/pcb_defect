import cv2
from PIL import Image, ImageOps
import numpy as np
import imutils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from keras.optimizers import SGD
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.models import load_model

def morph_transform(ref, test):
    img1 = test
    img2 = ref
    height, width, depth = img2.shape
    orb_detector = cv2.ORB_create(5000)
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches = list(matches)  # Convert matches to a list
    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    transformed_img = cv2.warpPerspective(test, homography, (width, height))
    return transformed_img



image1 = cv2.imread(r'C:\Users\rjeev\Desktop\project\data set\PCB-Defects-Classification-Using-Deep-Learning-main\Code\defect\2a.png')
image2 = cv2.imread(r'C:\Users\rjeev\Desktop\project\data set\PCB-Defects-Classification-Using-Deep-Learning-main\Code\not_defect\2.png')
ref_test = morph_transform(image1 , image2)
image2 = ref_test
image1 = cv2.medianBlur(image1,5)
image2 = cv2.medianBlur(image2,5)
image_res = cv2.bitwise_xor(image1 , image2)
image_res = cv2.medianBlur(image_res,5)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
image_res = cv2.morphologyEx(image_res, cv2.MORPH_CLOSE, kernel1)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
image_res = cv2.morphologyEx(image_res, cv2.MORPH_OPEN, kernel2)
thresh , image_res = cv2.threshold(image_res , 125 , 255 , cv2.THRESH_BINARY)
edges = cv2.Canny(image_res, 30, 200) 
cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
img2 = cv2.imread(r'C:\Users\rjeev\Desktop\project\data set\PCB-Defects-Classification-Using-Deep-Learning-main\Code\defect\2a.png')



X=[]
Y=[]
CX=[]
CY=[]
C=[]

for c in cnts:
	M = cv2.moments(c)
	if(M["m00"] != 0):
		cx = int(M["m10"] / M["m00"])
		cy = int(M["m01"] / M["m00"])
		CX.append(cx)
		CY.append(cy)
		C.append((cx,cy))




im = Image.open(r"C:\Users\rjeev\Desktop\project\data set\PCB-Defects-Classification-Using-Deep-Learning-main\Code\not_defect\2.png")
model = load_model(r'C:\Users\rjeev\Desktop\project\data set\PCB-Defects-Classification-Using-Deep-Learning-main\Code\pcb_defect_test.h5')
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# print(model.summary())
classes = {
  0: "Open",
  1: "Short",
  2: "Mousebite",
  3: "Spur",
  4: "Copper",
  5: "Pin-Hole"
}

pred = []
confidence = []
for c in C:
    im1 = im.crop((c[0] - 32, c[1] - 32, c[0] + 32, c[1] + 32))
    im1 = np.array(im1)
    im1 = cv2.resize(im1, (64, 64))  # Resize the image
    
    # Check if the image has 3 channels before converting to grayscale
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    
    im1 = np.expand_dims(im1, axis=-1)  # Add channel dimension
    im1 = np.expand_dims(im1, axis=0)   # Add batch dimension
    
    print(im1.shape)
    a = model.predict(im1, verbose=1, batch_size=1)
    pred.append(np.argmax(a))
    confidence.append(a)

# Rest of the code remains unchanged


plot_final = plt.imshow(img2)
plt.scatter(CX, CY, c='r', s=4)
for i, txt in enumerate(pred):
    if txt in classes:
        label = classes[txt]
        confidence_value = confidence[i][0][txt]
        plt.annotate([label, confidence_value], (CX[i], CY[i]), color='r')
print(pred)
plt.show()








