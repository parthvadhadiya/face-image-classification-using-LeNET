from sklearn.model_selection import train_test_split  
from keras.optimizers import SGD  
from keras.utils import np_utils  
import numpy as np  
import argparse  
import cv2  
import os  
import sys  
import keras
import argparse
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
 
file_path = "<set your file path here>"

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load_model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

img_size = 100
class LeNet:
	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):
		model = Sequential()
		model.add(Conv2D(20, (5, 5), padding="same",  
	                 input_shape=(width,height,depth)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(50,(5, 5), padding="same"))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation('relu'))



		model.add(Dense(classes))
		model.add(Activation('softmax'))


		if weightsPath is not None:
			model.load_weights(weightsPath)

		return model

people_folder = file_path
images = []

labels = []

labels_people = {}
people = [person for person in os.listdir(people_folder)]
for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + '/' + person):
            images.append(cv2.imread(people_folder +'/'+ person + '/' + image, 0))
            labels.append(i)

(trainData, testData, trainLabels, testLabels) = train_test_split(images, labels, test_size=0.10)
'''
if K.image_data_format() == 'channels_first':
	trainData = trainData.reshape(trainData.shape[0], 1, img_size, img_size)
	testData = testData.reshape(testData.shape[0], 1, img_size, img_size)
	input_shape = (1, img_size, img_size)
else:
	trainData = trainData.reshape(trainData.shape[0], img_size, img_size, 1)
	testData = testData.reshape(testData.shape[0], img_size, img_size, 1)
	input_shape = (img_size, img_size, 1)
'''
trainLabels = np_utils.to_categorical(trainLabels, 5)
testLabels = np_utils.to_categorical(testLabels, 5)

trainData = np.asarray(trainData)
testData = np.asarray(testData)

trainData = trainData.reshape(trainData.shape[0], img_size, img_size, 1)
testData = testData.reshape(testData.shape[0], img_size, img_size, 1)

trainData = trainData.astype('float32')
testData = testData.astype('float32')

trainData /= 255
testData /= 255

print(trainData[0].shape)
print("=================")
print(trainData.shape[0])
print("+++++++++++")
print(trainData.shape)
opt = SGD(lr=0.01)

model = LeNet.build(width=100, height=100, depth=1, classes=5, weightsPath=args["weights"] if args["load_model"] > 0 else None)
#model = LeNet.build(input=input_shape, classes=5, weightsPath=None)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

if args["load_model"] < 0:
	model.fit(trainData, trainLabels, batch_size=128, epochs=25, verbose=1)

(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


#saving model
model.save('model.h5')
print("model saved")

if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)
