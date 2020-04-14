#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten



# Loading F-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# One hot encoded outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

X_train = X_train / 255
X_test = X_test / 255



# Creating THE CNN
    
def CNN():
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='Adam')
    return model
    
model=CNN()

model.summary()


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

score, acc = model.evaluate(X_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)



