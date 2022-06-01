import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv3D, MaxPool2D, Flatten
import numpy as np

# ====== DAY - 1 ======
# 1. VGG16
vgg16 = Sequential()
vgg16.add(Conv2D(input_shape=(224,224),filters = 64, kernel_size = (3,3)
          ,padding = "same", activation = "relu"))
vgg16.add(Conv2D(filters = 64, kernel_size=(3,3),padding ="same",
                 activation = "relu"))
vgg16.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
vgg16.add(Conv2D(filters = 128, kernel_size = (3,3),padding = "same"
                 , activation = "relu"))
vgg16.add(Conv2D(filters = 128, kernel_size= (3,3), padding = "same",
                 activation = "relu"))
vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg16.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same",
                 activation = "relu"))
vgg16.add(Conv2D(filters = 256, kernel_size=(3,3), padding = "same",
                 activation= "relu"))
vgg16.add(Conv2D(filters = 256, kernel_size= (3,3), padding= "same",
                 activation= "relu"))
vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg16.add(Conv2D(filters=512, kernel_size = (3,3), padding = "same",
                 activation ="relu"))
vgg16.add(Conv2D(filters=512, kernel_size = (3,3), padding = "same",
                 activation ="relu"))
vgg16.add(Conv2D(filters=512, kernel_size = (3,3), padding = "same",
                 activation ="relu"))
vgg16.add(MaxPool2D(pool_size=(2,2), strides =(2,2)))
vgg16.add(Conv2D(filters=512, kernel_size = (3,3), padding = "same",
                 activation ="relu"))
vgg16.add(Conv2D(filters=512, kernel_size = (3,3), padding = "same",
                 activation ="relu"))
vgg16.add(Conv2D(filters=512, kernel_size = (3,3), padding = "same",
                 activation ="relu"))
vgg16.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))
vgg16.add(Flatten())
vgg16.add(Dense(units = 4096,activation="relu"))
vgg16.add(Dense(units = 4096, activation = "relu"))
vgg16.add(Dense(units = 2, activation="softmax"))

# DAY - 2
# ===== VGG19 =======

vgg19 = Sequential()
vgg19.add(Conv2D(input_shape=(224,224), filters= 64, kernel_size=(3,3),
                 padding="same",activation="relu"))
vgg19.add(Conv2D(filters=64, kernel_size=(3,3), padding="same",
                 activation="relu"))
vgg19.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
vgg19.add(Conv2D(filters = 128, kernel_size=(3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 128, kernel_size=(3,3),padding = "same",
                 activation = "relu"))
vgg19.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
vgg19.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same",
                 activation="relu"))
vgg19.add(Flatten())
vgg19.add(Dense(4096,activation = "relu"))
vgg19.add(Dense(4096,activation = "relu"))
vgg19.add(Dense(2, activation="softmax"))

