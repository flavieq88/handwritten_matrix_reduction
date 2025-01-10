# Digit Recognizer

This is a simple Convolutional Neural Network implemented with Tensorflow to classify images of handwritten digits. Then, the CNN is used to recognize matrices from images of handwritten digits.

The images are from the MNIST dataset.

![image](https://github.com/flavieq88/digit_recognizer/assets/166056837/a303fc15-f58a-4df1-acbb-51a8a03538af)
<br>Example of digit image and prediction

I used OpenCV to process images of multiple digits in a matrix format, through image preprocessing and contour detection to locate individual digits. Those individual digits are cropped into their own image of a similar format to the training dataset and fed into the classifier. Thus, the program is able to convert an image of digits into a matrix classified using machine learning, given the size of the matrix.

![image](https://github.com/user-attachments/assets/823da6f4-26b3-4304-9172-67c1b0ff18d2)

To do: the next step is to write a Gauss Jordan Elimination algorithm, so that the row reduced echelon form (RREF) of the given matrix is returned.

# Installation
Clone the repository. Then, activate a virtual environment and install the dependencies:
```
pip install -r requirements.txt
```
To run the matrix image converter, run `row_reduce_image.py` and it will convert example images from the `testing_images/multi_digit` directory. <br>
The model design can be found in  `classifier_model/digit_classifier.ipynb`.

## Model Results
The model achieved a 99% accuracy. I have also generated the confusion matrix.

![image](https://github.com/user-attachments/assets/87a65ed3-5cc4-487e-a7ed-96f397842969)


## References
I referenced a tutorial found on the Tensorflow website for the CNN architechture: https://www.tensorflow.org/tutorials/images/cnn.
