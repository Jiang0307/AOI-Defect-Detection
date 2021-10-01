import os
import ntpath
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from random import shuffle
from PIL import Image
from time import gmtime, strftime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers, regularizers, metrics, regularizers, models, layers, utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, SpatialDropout2D
from keras.models import Sequential
from keras.applications import VGG16, ResNet50

trainCSV = pd.read_csv("train.csv")

train_path = 'train_images/train/'
valid_path = 'train_images/valid/'
test_path = 'train_images/test/'
test_path2 = 'test_images/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

#conv_base = V(weights="imagenet",include_top=False,input_shape=(224,224,3))


def build_model():
    selected = ResNet50(include_top=False, weights='imagenet',
                        input_shape=(224, 224, 3))
    model = Sequential()
    model = Sequential()
    model.add(selected)
    model.add(GlobalAveragePooling2D())
    # model.add(Dropout(0.9))
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(1024, activation='sigmoid'))
    # model.add(Dropout(0.7))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    selected_optimizer = Adam(learning_rate=0.00001)
    #selected_optimizer = RMSprop(lr=0.000001)
    model.compile(optimizer=selected_optimizer,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


model = build_model()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//validation_generator.batch_size
)
model.save('model/VGG16_pretrain_all_' +
           strftime("%Y_%m_%d_%H_%M_%S", gmtime())+".model")
print('model/VGG16_pretrain_all_'+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+".model")


test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(
    test_generator,
    steps=test_generator.samples//test_generator.batch_size)

print('test acc:', test_acc)
