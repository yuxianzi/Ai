import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt

labels = ['flower', 'triangle', 'cloud', 'bicycle', 'sun', 't-shirt', 'microphone', 'door', 'ladder', 'apple']

# 调用 keras 接口读取模型
# （1）补充 ↓
model = tf.keras.models.load_model('model.h5')
model.summary()
# （1）补充 ↑

# 使用 opencv 库读取图像并对其正规化操作，已知图像每个像素点的值不超过 255，要求将其正规化到 [0~1] 范围内
# （2）补充 ↓
image = cv.imread('test.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = image.astype(np.float32) / 255.0
print(image)
# （2）补充 ↑


# 使用 plt 显示数据集
# （3）补充 ↓
plt.imshow(image, cmap='gray')
plt.show()
# （3）补充 ↑

image = np.reshape(image,[28,28,1])
image = np.expand_dims(image, axis=0)

# 调用 keras 接口完成预测，并与标签集合，一一对应
# （4）补充 ↓
result = model.predict(image)
print(result)
jsonData = {}
for (label,idx) in zip(labels,range(len(labels))):
    jsonData[label]=result[0][idx]
print(jsonData)
# （4）补充 ↑