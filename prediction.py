# Starter code for CS 165B HW4
# coding: utf-8
# https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
"""
Implement the testing procedure here.

Inputs:
    Given the folder named "hw4_test" that is put in the same directory of your "predictio.py" file, like:
    - Main folder
        - "prediction.py"
        - folder named "hw4_test" (the exactly same as the uncompressed hw4_test folder in Piazza)
    Your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes:
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab,
        elsewise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files,
    please use open('./necessary.file') instaed of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 socre for your hw4.
"""

import numpy as np
import sys
import os
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model

def readImage(filename):
    image = Image.open(filename)
    return np.asarray(image.getdata())

def loadTraining():
    trainingImages = []
    trainingLabels = []
    numOfClasses = 10
    numOfImagesInClasses = 1000

    for i in range(numOfClasses):
        for j in range(numOfImagesInClasses):
            label = [0] * numOfClasses
            label[i] = 1
            label = [float(k) for k in label]
            image = readImage('./hw4_train/' + str(i) + '/' + str(i) + '_' + str(j) + '.png')
            image = [float(k) for k in image]
            trainingImages.append(image)
            trainingLabels.append(label)
    trainingImages, trainingValidationImages, trainingLabels, trainingValidationLabels = train_test_split(trainingImages, trainingLabels, test_size=0.1)
    return(np.array(trainingImages), np.array(trainingLabels), np.array(trainingValidationImages), np.array(trainingValidationLabels))

def loadTesting():
    testingImages = []

    for i in range(10000):
        image = readImage('./hw4_test/' + str(i) + '.png')
        image = [float(k) for k in image]
        testingImages.append(image)
    return np.array(testingImages)

trainingImages, trainingLabels, trainingValidationImages, trainingValidationLabels = loadTraining()
testingImages = loadTesting()

trainingImages = trainingImages.reshape([-1, 28, 28, 1])
trainingValidationImages = trainingValidationImages.reshape([-1, 28, 28, 1])
testingImages = testingImages.reshape([-1, 28, 28, 1])
trainingImages = trainingImages.astype('float32') / 255
trainingValidationImages = trainingValidationImages.astype('float32') / 255
testingImages = testingImages.astype('float32') / 255

model = tf.keras.Sequential()

if os.path.isfile('keras_model'):
    model = load_model('keras_model')
else:
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    #model.summary()

    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    model.fit(trainingImages,
             trainingLabels,
             batch_size=64,
             epochs=10,
             validation_data=(trainingValidationImages, trainingValidationLabels))
    model.save('keras_model')

#score = model.evaluate(testingImages, y_test, verbose=0)
#print('\n', 'Test accuracy:', score[1])
prediction = np.array(model.predict(testingImages))
prediction = np.argmax(prediction, axis=1)

fstream = open('prediction.txt', 'w+')
for i in range(len(prediction)):
    fstream.write(str(prediction[i]) + '\n')
fstream.close()
