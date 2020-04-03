# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:41:15 2020

@author: Xuda Lin
"""
import os
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd
from PIL import Image
from numpy import array
import numpy as np

def createDataMatrix(images):
	print("Creating data matrix",end=" ... ")
	''' 
	Allocate space for all images in one data matrix. 
        The size of the data matrix is
        ( w  * h  * 3, numImages )
       
        where,
        
        w = width of an image in the dataset.
        h = height of an image in the dataset.
        3 is for the 3 color channels.
        '''
  
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	
	print("DONE")
	return data

def createNewFace(*args):
	# Start with the mean image
	output = averageFace
	
	# Add the eigen faces with the weights
	for i in range(0, NUM_EIGEN_FACES):
		'''
		OpenCV does not allow slider values to be negative. 
		So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
		''' 
		sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
		weight = sliderValues[i] - MAX_SLIDER_VALUE/2
		output = np.add(output, eigenFaces[i] * weight)

	# Display Result at 2x size
	output = cv2.resize(output, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

# Set path
# Path for all images:
#dataset_path = "./lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/"
# Path for two people with more than 10 images:
dataset_path="./Test/"
print("Number of sub folder in images folder =",len(os.listdir(dataset_path)))

# Load images and label persons
dataset = []
for path in glob.iglob(os.path.join(dataset_path, "**", "*.jpg")):
    person = path.split("\\")[-2]
    image=np.asarray(cv2.imread(path))
    dataset.append({"person":person, "path": path,"image":image})

# Convert to pd.DataFrame and keep the persons with more than 10 images
dataset = pd.DataFrame(dataset)
dataset = dataset.groupby("person").filter(lambda x: len(x) > 10)
dataset.head(10)

# Number of EigenFaces
NUM_EIGEN_FACES = 10
 
# Maximum weight
MAX_SLIDER_VALUE = 255

# Convert images to nparray
imagenumber=len(dataset)
images=np.zeros((imagenumber,250,250,3),dtype=np.float32)
i=0
for index,row in dataset.iterrows():
#    print("index:",index)
#    print("image:",row['image'])
    images[i,:,:,:]=np.asarray(row['image'])
    i+=1

#print(images[0])

# Size of images
sz = images[0].shape

# Create data matrix for PCA.
data = createDataMatrix(images)

# Compute the eigenvectors from the stack of images created
print("Calculating PCA ", end="...")
mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
print ("DONE")
averageFace = mean.reshape(sz)

eigenFaces = [];

for eigenVector in eigenVectors:
    eigenFace = eigenVector.reshape(sz)
    eigenFaces.append(eigenFace)
 

# Create window for displaying Mean Face
cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

# Display result at 2x size
output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
cv2.imshow("Result", output)

# Create Window for trackbars
cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)
sliderValues = []

# Create Trackbars
for i in range(0, NUM_EIGEN_FACES):
    sliderValues.append(int(MAX_SLIDER_VALUE/2))
    cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)

# You can reset the sliders by clicking on the mean image.
#cv2.setMouseCallback("Result", resetSliderValues);
#print('''Usage:  Change the weights using the sliders.Click on the result window to reset sliders.Hit ESC to terminate program.''')

cv2.waitKey(0)
cv2.destroyAllWindows()
