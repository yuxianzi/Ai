import os
import glob
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocess import preprocess as pp

x_train, y_train, x_test, y_test, class_names = pp.getData()

# Define model
model = keras.Sequential()

# 按照给定的模型结构，排列正确的模型建立代码
# （2）排列 ↓
model.add(layers.Convolution2D(16, (3, 3),
    padding='same',
    input_shape=x_train.shape[1:], activation='relu', name="conv_1"))
model.add(layers.MaxPooling2D(pool_size=(2,2), name="maxpool_1"))
model.add(layers.Convolution2D(32, (3 ,3), padding='same', activation= 'relu', name='conv_2'))
model.add(layers.MaxPooling2D(pool_size=(2,2), name="maxpool_2"))
model.add(layers.Convolution2D(64, (3 ,3), padding='same', activation= 'relu', name='conv_3'))
model.add(layers.MaxPooling2D(pool_size=(2,2), name="maxpool_3"))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', name="dense_1", kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu', name="dense_2", kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax', name="dense_3"))
# （2）排列 ↑

# Train model
adam = tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['categorical_accuracy'])

#（3）输出模型结构 ↓
model.summary()
#（3）输出模型结构 ↑

import datetime
log_dir="answer/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 设置 Tensorboard 接口回调 ↓
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# 设置 Tensorboard 接口回调 ↑

# 使用给定的数据集训练模型并导出，其中验证比例为 0.1，每批大小为 256，训练 10 个纪元
# 训练并导出模型 ↓
model.fit(x = x_train, y = y_train, validation_split=0.1, batch_size= 256, verbose=2, epochs=10, callbacks=[tensorboard_callback])
model.save('answer/model.h5')
# 训练并导出模型 ↑

