import keras
from keras.models import load_model
import cv2
import numpy as np
from keras.optimizers import SGD

model = load_model('model.h5')

opt = SGD(lr=0.01)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

img = cv2.imread('15.jpg')
img = cv2.resize(img,(100,100))
img = np.reshape(img,[-1,100,100,1])

classes = model.predict_classes( img, batch_size=128, verbose=0)
#classes = model.predict(img)
#classes = model.predict_classes(img)

print(classes)
