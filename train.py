import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pathlib
from sklearn.preprocessing import StandardScaler
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = pathlib.Path("train")
test_data_dir = pathlib.Path("test")


train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# đọc ảnh từ file train đưa về kích cở 48x 48
train_generator = train_data_gen.flow_from_directory(
        train_data_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# đọc ảnh từ file train đưa về kích cở 48x 48
validation_generator = validation_data_gen.flow_from_directory(
        test_data_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
#lop cnn
emotion_model = Sequential([
    ##cnn 1
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    #cnn 2
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    #cnn3
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

emotion_model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

emotion_model.summary()
# Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,#số ảnh train trong luot train
        epochs=30,#
        validation_data=validation_generator,
        validation_steps=7178 // 64)

emotion_model.save('model.h5')