# to process an image of handwritten digits, to be able to be fed through the digit classifier

import cv2
import keras

model = keras.models.load_model("classifier_model/model.keras")
