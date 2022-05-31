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


