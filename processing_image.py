# to process an image of handwritten digits, to be able to be fed through the digit classifier

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def processImage(filepath):
    """Returns the processed image that can be fed into the classifier"""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(255-image, (56, 56)) #invert colours and make it smaller
    image = cv2.dilate(image, np.ones((3,3), np.uint8), iterations=1) #thicken the lines
    #thresholds so that "white" becomes true white and "dark" becomes true black and filter out some noise 
    (thresh, image) = cv2.threshold(image, 130, 255, cv2.ADAPTIVE_THRESH_MEAN_C) #mostly useful for paper photos
    image = cv2.resize(image, (28, 28)) #invert colours and make it 28x28
    #doing the resizing in parts makes the final image smoother
    image = image/255 #normalize pixel value ranges to be 0-1

    #processing images to make them shaped more similar to the MNIST dataset: digit occupies center 20x20 pixels

    #cut down until get only the number (no empty space around)
    while np.sum(image[0]) == 0: #get rid of all empty top rows
        image = image[1:]
    while np.sum(image[:,0]) == 0: #get rid of all empty left columns
        image = np.delete(image,0,1)
    while np.sum(image[-1]) == 0: #get rid of all empty bottom rows
        image = image[:-1]
    while np.sum(image[:,-1]) == 0: #get rid of all empty right columns
        image = np.delete(image,-1,1)

    #now scale it to 20x20
    rows,cols = image.shape 
    if rows > cols: 
        factor = 20/rows
        rows = 20
        cols = int(round(cols*factor))
        image = cv2.resize(image, (cols,rows)) 
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        image = cv2.resize(image, (cols, rows))
    
    #pad all around to have black surroundings and get 28x28 image
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    image = np.lib.pad(image,(rowsPadding,colsPadding),'constant')
    return image

def predictImage(filepath, classifier):
    """Returns a predicted label of the image at specified filepath, given by the classifier model"""
    image = processImage(filepath)    
    pred = classifier.predict(np.array([image,]))
    return np.argmax(pred)

#testing code
if __name__ == "__main__":
    
    #have all the testing images in one list
    ls = [] #store the file names
    predictions = [] #store the predictions

    directory = "testing_images"
    for filename in os.listdir(directory):
        ls.append(filename)

    #load the classifier 
    model = keras.models.load_model("classifier_model/model1.keras")
    plt.figure(figsize=(11,7))
    for i in range(len(ls)):
        x = predictImage(directory+"/"+ls[i], model)
        predictions.append(x)
        plt.subplot(3, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(processImage(directory+"/"+ls[i]))
        plt.xlabel(f"Predicted {x}")
    plt.show()

    #calculate accuracy
    correct = 0
    for i in range(len(predictions)):
        if str(predictions[i]) == ls[i][0]:
            correct += 1
    print(f"Accuracy = {100*correct/len(ls)}%")