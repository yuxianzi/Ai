import os
import glob
import numpy as np
import cv2 as cv
from tensorflow import keras

all_files = glob.glob(os.path.join('./data', '*.npy'))
x = np.empty([0, 784])
y = np.empty([0])
class_names = []

for idx, file in enumerate(all_files):
    # 使用 numpy 读取每一个数据集文件
    # （1）补充 ↓
    data = np.load(file)
    # （1）补充 ↑

    labels = np.full(data.shape[0], idx)

    # 把所有类型的数据集合并到同一个变量中
    # （1）补充 ↓
    x = np.concatenate((x, data), axis=0)
    # （1）补充 ↑

    y = np.append(y, labels)

    class_name, ext = os.path.splitext(os.path.basename(file))
    class_names.append(class_name)

    # 每种图片各取第一张保存在 answer 目录下
    # （2）补充 ↓
    oneImage = data[0]
    oneImage = np.reshape(oneImage, [28, 28, 1])
    cv.imwrite("answer/" + file[7:-4] + ".jpg", oneImage)
    # （2）补充 ↑

# 输出数据集中图片总数
# （1）补充 ↓
print(x.shape[0])
# （1）补充 ↑

# 已知每张图像为 28*28 的单通道灰度图，并且每个像素点的值不超过 255，要求将图像的像素点值正规化到 [0~1] 区间，并输出第一组数据集的图像存储和标签存储，以及总数据集（包括图像和标签）在处理完之后的 Shape
#（3）补充 ↓
x = x.reshape(x.shape[0],28,28,1);
x /= 255
y = keras.utils.to_categorical(y,5)
print(x[0])
print(y[0])
print(x.shape)
print(y.shape)
#（3）补充 ↑

data = None
labels = None

permutation = np.random.permutation(y.shape[0])
x = x[permutation, :]
y = y[permutation]

# 设置比例，计算 20% 数据集的数量
#（4）补充 ↓
vfold_size = int(x.shape[0]*0.2)
#（4）补充 ↑

x_test = x[0:vfold_size, :]
y_test = y[0:vfold_size]

x_train = x[vfold_size:x.shape[0], :]
y_train = y[vfold_size:y.shape[0]]

# 将图像和标签分别导出成trainImagedataset.npy、trainLlabel.npy、testImage.npy、testLabel
#（4）补充 ↓
np.save("trainImagedataset.npy",x_train)
np.save("trainLlabel.npy",y_train)
np.save("testImage.npy",x_test)
np.save("testLabel.npy",y_test)
#（4）补充 ↑
