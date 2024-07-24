import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def processImage(image):
    """Returns the processed image that can be fed into the classifier"""
    image = cv2.resize(255-image, (56, 56)) #invert colours and make it smaller
    image = cv2.dilate(image, np.ones((2,2), np.uint8), iterations=2) #thicken the lines
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

    #now scale it to 20x20px
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
    """Returns a predicted labels and processed images of the digits of file at specified filepath, given by the classifier model"""
    images = contours(filepath)
    results = []
    processed_images = []
    
    for image in images:
        processed = processImage(image) 
        processed_images.append(processed)
        pred = classifier.predict(np.array([processed,]))
        results.append(np.argmax(pred))

    return results, processed_images


def contours(filepath):
    """Returns a list of all contours of an image"""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    #thresholds so that "white" becomes true white and "dark" becomes true black and filter out some noise 
    (thresh, gray) = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY) #mostly useful for paper photos

    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1) #thicken lines
    flooded = gray.copy()
    height, width = gray.shape[:2] 
    mask = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(flooded, mask, (0, 0), 0)

    #combine with original image such that places that were black originally become white
    flooded[np.where(gray==0)]=255
    
    #find all the contours within image
    contours, hierarchy = cv2.findContours(flooded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #get rectangles for countours and remove contours that are enclosed in anotehr
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    
    #take the small recatangles as their own image
    images = [image[y:y+h, x:x+w] for x, y, w, h in rectangles]

    #need to square of fthe images for preprocessing
    for i, image in enumerate(images): 
        rows, cols = image.shape
        if rows>cols:
            colsPadding = (int(math.ceil((rows-cols)/2.0)), int(math.floor((rows-cols)/2.0)))
            images[i] = np.pad(image,  ((0,0), colsPadding), mode="constant", constant_values=255) #pad sides with white
        else:
            rowsPadding = (int(math.ceil((cols-rows)/2.0)), int(math.floor((cols-rows)/2.0)))
            images[i] = np.pad(image,  (rowsPadding, (0,0)), mode="constant", constant_values=255) #pad sides with white
        
    return images
        

if __name__ == "__main__":
    ls = [] #store the file names

    directory = "testing_images/multi_digit"
    for filename in os.listdir(directory):
        ls.append(filename)
    
    model = keras.models.load_model("classifier_model/model.keras")
    
    for filename in ls:
        results, images = predictImage(directory+"/"+filename, model)
        #plot original image
        plt.figure(figsize=(12, 8))
        plt.title(filename)
        
        plt.subplot(2, 5, 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(cv2.imread(directory+"/"+filename))
        
        for i in range(len(results)):
            plt.subplot(2, 5, i+2)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i])
            plt.xlabel(f"Predicted {results[i]}")
        plt.show()
