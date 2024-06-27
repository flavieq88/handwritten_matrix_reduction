# to process an image of handwritten digits, to be able to be fed through the digit classifier

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import os

def predictImage(filepath, classifier):
    """Reads the image at the specified filepath and returns a predicted label of the digit found by the classifier model"""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(255-image, (28, 28)) #invert colours and make it 28x28
    (thresh, image) = cv2.threshold(image, 5, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    image = image/255 #normalize pixel value ranges to be 0-1
    pred = classifier.predict(np.array([image,]))
    plt.figure()
    plt.xticks()
    plt.yticks()
    plt.imshow(image)
    plt.xlabel(f"True label: {filepath.strip("testing_images/").strip(".png")}\n Predicted label: {np.argmax(pred)}")
    plt.show()
    return np.argmax(pred)

if __name__ == "__main__":
#have all the testing images in one list
    ls = [] #store the file names
    predictions = [] #store the predictions
    directory = "testing_images"
    for filename in os.listdir(directory):
        ls.append(filename)

    #load the classifier 
    model = keras.models.load_model("classifier_model/model.keras")

    for file in ls:
        predictions.append(predictImage(directory+"/"+file, model))

    print(predictions)
