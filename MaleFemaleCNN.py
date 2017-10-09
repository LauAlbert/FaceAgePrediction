import numpy as np
import cv2
import glob
import os
#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
#img = cv2.imread('C:\\Users\\Albert-Desktop\\.spyder-py3\\Lenna.png')
#grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#faces = faceCascade.detectMultiScale(grayScale, 1.3, 5)
#for (x,y,w,h) in faces:
#    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = grayScale[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]
#
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#s = "C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd1\\data\\smaller\\*\\*fa.ppm"
#for filename in glob.glob(s): #assuming gif
#    img = cv2.imread(filename)
#    print(img.shape)
#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

#file = "C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd1\\data\\ground_truths\\name_value\\00001\\00001.txt"
#faceDic = {}
#with open(file) as f:
#    data = f.readlines()
#    for d in data:
#        key, value = d.rstrip().split("=")
#        faceDic[key] = value
#    print(1)





mainDir = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd1\\data'
valueDir = 'ground_truths\\name_value'
imgDir = 'thumbnails'
#
#import os
#
#root = os.path.join(mainDir, valueDir)
#for item in os.listdir(root):
#    print(item)
##    if os.path.isfile(os.path.join(root, item)):
#        print item







xTrain=[]
yTrain=[]

#for root, dirs, filenames in os.walk(mainDir + '\\' + valueDir):


root = os.path.join(mainDir, valueDir)
for item in os.listdir(root):
#    if(len(xTrain) >10):
#        break;
#    print(os.path.split(root)[1])
    s = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd1\\data\\' + imgDir  + '\\'+ item +'\\*fa.ppm'
    file = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd1\\data\\ground_truths\\name_value\\' + item + '\\' + item + '.txt'
    faceDic = {}
    with open(file) as f:
        data = f.readlines()
        for d in data:
            key, value = d.rstrip().split("=")
            faceDic[key] = value
    if(faceDic['gender'] == 'Male'):
        yTrain.append(1)
    else:
        yTrain.append(0)

    if (len(glob.glob(s)) == 0):
        s = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd1\\data\\' + imgDir  + '\\'+ item +'\\*fa_a.ppm'
#        print(os.path.split(root)[1])
    for filename in glob.glob(s): #assuming gif
        img = cv2.imread(filename)

        xTrain.append(img)
        break;
    if (len(yTrain) != len(xTrain)):
        print(os.path.split(root))
        break;
        
        

mainDir = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd2\\data'
valueDir = 'ground_truths\\name_value'
imgDir = 'thumbnails'
#
#import os
#
#root = os.path.join(mainDir, valueDir)
#for item in os.listdir(root):
#    print(item)
##    if os.path.isfile(os.path.join(root, item)):
#        print item







xTest=[]
yTest=[]

#for root, dirs, filenames in os.walk(mainDir + '\\' + valueDir):


root = os.path.join(mainDir, valueDir)
for item in os.listdir(root):
#    if(len(xTrain) >10):
#        break;
#    print(os.path.split(root)[1])
    s = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd2\\data\\' + imgDir  + '\\'+ item +'\\*fa.ppm'
    file = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd2\\data\\ground_truths\\name_value\\' + item + '\\' + item + '.txt'
    faceDic = {}
    with open(file) as f:
        data = f.readlines()
        for d in data:
            key, value = d.rstrip().split("=")
            faceDic[key] = value
    if(faceDic['gender'] == 'Male'):
        yTest.append(1)
    else:
        yTest.append(0)

    if (len(glob.glob(s)) == 0):
        s = 'C:\\Users\\Albert-Desktop\\colorferet\\colorferet\\dvd2\\data\\' + imgDir  + '\\'+ item +'\\*fa_a.ppm'
#        print(os.path.split(root)[1])
    for filename in glob.glob(s): #assuming gif
        img = cv2.imread(filename)

        xTest.append(img)
        break;
    if (len(yTest) != len(xTest)):
        print(os.path.split(root))
        break;
        
        
        
imgHeight = 192
imgWidth = 128
#imgHeight = 384
#imgWidth = 256
#imgHeight = 768
#imgWidth = 512
xTrain = np.reshape(xTrain, (len(xTrain), imgHeight, imgWidth, 3))
xTest = np.reshape(xTest, (len(xTest), imgHeight, imgWidth, 3))
#        cv2.imshow('image',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#    
#
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import np_utils

classifier = Sequential()

classifier.add(Convolution2D(32, kernel_size=(3, 3), border_mode='same', input_shape=(imgHeight, imgWidth, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, kernel_size=(3, 3), border_mode='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units=1024, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer=keras.optimizers.Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
epochs = 12
batch_size=128

num_classes = 2

#yTrain = np_utils.to_categorical(yTrain, num_classes)


#classifier.fit(xTrain, 
#               yTrain,
#               batch_size=128,
#               epochs=12,
#               verbose=1,
#               validation_data=(xTrain, yTrain))
#y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)



# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(xTrain)

# fits the model on batches with real-time data augmentation:
classifier.fit_generator(datagen.flow(xTrain, yTrain, batch_size=batch_size),
                    steps_per_epoch=len(xTrain) / 32, epochs=epochs, validation_data=(xTest, yTest))

## here's a more "manual" example
#for e in range(epochs):
#    print('Epoch', e)
#    batches = 0
#    for x_batch, y_batch in datagen.flow(xTrain, yTrain, batch_size=32):
#        classifier.fit(x_batch, y_batch)
#        batches += 1
#        if batches >= len(xTrain) / 32:
#            # we need to break the loop by hand because
#            # the generator loops indefinitely
#            break
#
#import keras
#from keras.datasets import cifar10
#from keras.utils import np_utils
#
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#y_train = np_utils.to_categorical(y_train, 10)
#y_test = np_utils.to_categorical(y_test, num_classes)