#!~/.conda/envs python
# -*- coding: utf-8 -*-
from core.utils.captchagenerator import create, show
import tensorflow as tf
import functools  # 来源Python标准库的包，用于处理函数和可调用对象的工具
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as opts
import core.utils.network as net
import matplotlib.pyplot as pyp
import numpy as np
import PIL.Image as Image
from tensorflow.python.keras.engine import data_adapter
import matplotlib.pyplot as plt
import time
import os
tf.config.experimental_run_functions_eagerly(True)
np.set_printoptions(suppress=True, threshold=np.inf)  # 设置不用科学计数法

BOX_COUNT = 3
CLASS_COUNT = 10
IMG_SHAPE = (56, 112, 1)
TAR_SHAPE = (4, 8, BOX_COUNT*5+CLASS_COUNT)


NETWORK = [
    layers.Input(shape=IMG_SHAPE, name="input"),
    layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding='same',
        activation=tf.nn.relu, name='Conv-1'
    ),
    layers.MaxPool2D(),
    net.RepeatConv2D(
        arg_lists=[
            {
                'filters': 64, 'kernel_size': (1, 1), 'padding': 'same',
                'activation': tf.nn.relu, 'name': 'Conv-2-1a-%d'
            },
            {
                'filters': 128, 'kernel_size': (3, 3), 'padding': 'same',
                'activation': tf.nn.relu, 'name': 'Conv-2-1b-%d'
            }
        ],
        repeat=1
    ),
    layers.Conv2D(
        filters=128, kernel_size=(1, 1), padding='same',
        activation=tf.nn.relu, name='Conv-2-2'
    ),
    layers.Conv2D(
        filters=256, kernel_size=(3, 3), padding='same',
        activation=tf.nn.relu, name='Conv-2-3'
    ),
    layers.MaxPool2D(),


    net.RepeatConv2D(
        arg_lists=[
            {
                'filters': 128, 'kernel_size': (1, 1), 'padding': 'same',
                'activation': tf.nn.relu, 'name': 'Conv-3-1a-%d'
            },
            {
                'filters': 256, 'kernel_size': (3, 3), 'padding': 'same',
                'activation': tf.nn.relu, 'name': 'Conv-3-1b-%d'
            }
        ],
        repeat=4
    ),
    layers.Conv2D(
        filters=256, kernel_size=(1, 1), padding='same',
        activation=tf.nn.relu, name='Conv-3-2'
    ),
    layers.Conv2D(
        filters=512, kernel_size=(3, 3), padding='same',
        activation=tf.nn.relu, name='Conv-3-3'
    ),
    layers.MaxPool2D(),


    net.RepeatConv2D(
        arg_lists=[
            {
                'filters': 256, 'kernel_size': (1, 1), 'padding': 'same',
                'activation': tf.nn.relu, 'name': 'Conv-4-1a-%d'
            },
            {
                'filters': 512, 'kernel_size': (3, 3), 'padding': 'same',
                'activation': tf.nn.relu, 'name': 'Conv-4-1b-%d'
            }
        ],
        repeat=2
    ),
    layers.Conv2D(
        filters=512, kernel_size=(1, 1), padding='same',
        activation=tf.nn.relu, name='Conv-4-2'
    ),
    layers.Conv2D(
        filters=1024, kernel_size=(3, 3), padding='same',
        activation=tf.nn.relu, name='Conv-4-3'
    ),
    layers.MaxPool2D(),

    layers.Conv2D(
        filters=1024, kernel_size=(3, 3), padding='same',
        activation=tf.nn.relu, name='Conv-5-1'
    ),
    layers.Conv2D(
        filters=1024, kernel_size=(3, 3), padding='same',
        activation=tf.nn.relu, name='Conv-5-2'
    ),
    layers.Flatten(),
    layers.Dense(units=800, activation=tf.nn.relu),
    layers.Reshape(target_shape=TAR_SHAPE),
]


def init_data_single(labels):  # 生成单一数据
    cell_h = IMG_SHAPE[0] / TAR_SHAPE[0]  # Cell的高度
    cell_w = IMG_SHAPE[1] / TAR_SHAPE[1]  # Cell的宽度
    labels_ = []
    for label in labels:  # 遍历所有物体
        x, y, w, h, class_ = label  # 解包label元组，x、y、w、h全部以像素为单位
        y_index = int((y+h/2)/cell_h)  # Cell列标
        x_index = int((x+w/2)/cell_w)  # Cell行标

        # y坐标加上一半高度算出中心y坐标，减去上面Cell总高度，算出距离当前Cell上边多少像素，除以Cell高度得到中心距离当前Cell上边百分之多少
        y = (y+h/2-cell_h*y_index)/cell_h

        # x坐标加上一半宽度算出中心x坐标，减去左面Cell总宽度，算出距离当前Cell左边多少像素，除以Cell宽度得到中心距离当前Cell左边百分之多少
        x = (x+w/2-cell_w*x_index)/cell_w

        w, h = w/cell_w, h/cell_h  # 求相对于Cell的宽度和高度

        if BOX_COUNT != 3:  # 防止上面改了下面不改
            raise ValueError('调整 Bonding Box 数目。')
        if w/h > 8/7:  # 三个Box中哪个来识别物体
            box_index = 0  # Box的序号
        elif w/h < 7/8:
            box_index = 2
        else:
            box_index = 1
        labels_.append((x_index, y_index, box_index, x, y, w, h, class_))  # 打包
    # 真实值，但是这里面包含的很多没用的nan,这些nan单纯用来占位，什么都不做
    targets = np.full(TAR_SHAPE, np.nan, dtype=float)

    for yi_ in range(TAR_SHAPE[0]):  # 遍历行
        for xi_ in range(TAR_SHAPE[1]):  # 遍历列
            for bi_ in range(BOX_COUNT):  # 遍历Box
                targets[yi_, xi_, bi_*5+4] = 0  # 绝大多数box置信度都是0
    for label in labels_:  # 遍历所有物体
        xi, yi, bi, x, y, w, h, class_ = label  # 解包label元组，全部相对位置相对尺寸
        vals = (x, y, w, h, 1.)  # 打包省代码，其中1.为置信度
        # 设置box尺寸真实值
        for i in range(5):  # 遍历x,y,w,h,c五个值
            targets[yi, xi, bi*5+i] = vals[i]  # 设置每一个真实值
        # 设置类别
        for ci in range(CLASS_COUNT):  # 遍历类别
            if ci == class_:  # 是物体真实类别
                targets[yi, xi, BOX_COUNT*5+ci] = 1.  # 正确类别的真实值是1
            else:  # 不是物体真实类别
                targets[yi, xi, BOX_COUNT*5+ci] = 0.  # 错误类别应该是0
    # print(labels_)
    return targets  # , mask


def init_data(size: int):
    imgs, targets = [], []
    for _ in range(size):
        img, labels = create()  # 创建验证码，img没有颜色深度，labels见下文
        target = init_data_single(labels)
        imgs.append(img)
        targets.append(target)
    return np.array(imgs), np.array(targets)


def y_hat2labels(y_hat):
    cell_h = IMG_SHAPE[0] / TAR_SHAPE[0]  # Cell的高度
    cell_w = IMG_SHAPE[1] / TAR_SHAPE[1]  # Cell的宽度
    labels = []
    for yi in range(TAR_SHAPE[0]):  # 遍历行
        for xi in range(TAR_SHAPE[1]):  # 遍历列
            for bi in range(BOX_COUNT):  # 遍历Box
                if y_hat[yi, xi, bi*5+4] >= .5:  # 负责预测的box
                    print(y_hat[yi, xi])
                    x = xi+y_hat[yi, xi, bi*5+0]-y_hat[yi, xi, bi*5+2]/2
                    # 计算中心相对x坐标
                    y = yi+y_hat[yi, xi, bi*5+1]-y_hat[yi, xi, bi*5+3]/2
                    # 计算中心相对y坐标
                    x = x*cell_w  # x
                    y = y*cell_h  # y
                    labels.append(
                        (
                            x, y,
                            y_hat[yi, xi, bi*5+2]*cell_w,  # w
                            y_hat[yi, xi, bi*5+3]*cell_h,  # h
                            np.argmax(y_hat[yi, xi, -CLASS_COUNT:])  # class
                        )
                    )
    return labels


def test(model):
    cell_h = IMG_SHAPE[0] / TAR_SHAPE[0]  # Cell的高度
    cell_w = IMG_SHAPE[1] / TAR_SHAPE[1]  # Cell的宽度
    image, _ = create()  # 创建验证码，img没有颜色深度，labels见下文
    img = np.reshape(image, IMG_SHAPE)
    y_hat = model(tf.convert_to_tensor([img]), training=False)
    y_hat = y_hat[0]
    labels = []
    for yi in range(TAR_SHAPE[0]):  # 遍历行
        for xi in range(TAR_SHAPE[1]):  # 遍历列
            for bi in range(BOX_COUNT):  # 遍历Box
                if y_hat[yi, xi, bi*5+4] >= .05:  # 负责预测的box
                    print(y_hat[yi, xi])
                    x = xi+y_hat[yi, xi, bi*5+0]-y_hat[yi, xi, bi*5+2]/2
                    # 计算中心相对x坐标
                    y = yi+y_hat[yi, xi, bi*5+1]-y_hat[yi, xi, bi*5+3]/2
                    # 计算中心相对y坐标
                    x = x*cell_w  # x
                    y = y*cell_h  # y
                    labels.append(
                        (
                            x, y,
                            y_hat[yi, xi, bi*5+2]*cell_w,  # w
                            y_hat[yi, xi, bi*5+3]*cell_h,  # h
                            np.argmax(y_hat[yi, xi, -CLASS_COUNT:])  # class
                        )
                    )
    image_, title = show(image, labels)
    image_.save('./export/imgs/%d_%s.png' %
                (int(time.time()*1e7), title), 'png')
    # print(labels)
    # plt.imshow(image_)
    # plt.title(title)
    # plt.show()
# @tf.function  # 这个注解控制将方法加入内存以提高运行速度


def yolo_loss(y, y_hat):
    tmp = y-y_hat
    zeros = tf.zeros_like(y_hat, dtype=tf.float32)
    loss = tf.where(tf.math.is_nan(y), x=zeros, y=tmp)
    return -tf.math.reduce_mean(loss)


def step(self, data):  # 定义训练步骤方法，模型每轮拟合都进行
    x, y_raw, _ = data_adapter.unpack_x_y_sample_weight(data)  # 把元组拆成单个值
    # with open('./export/log.log', 'a') as f:
    #     f.write('%s\n\n@\n\n' % self.trainable_variables)
    y_h = self(x, training=False)  # 非训练进行一次计算，得到模型当前参数输出
    # print(y_raw)
    y = tf.where(tf.math.is_nan(y_raw), x=y_h, y=y_raw)
    # print(y)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(self.trainable_variables)  # 监控网络参数进行过的运算
        y_hat = self(x, training=True)  # 得出模型输出
        loss = self.loss(y, y_hat)  # 计算模型损失
    grads = tape.gradient(loss, self.trainable_variables)  # GradientTape 自动求导
    self.optimizer.apply_gradients(  # 优化网络参数
        zip(grads, self.trainable_variables)  # 打包为元组
    )

    # with open('./export/log.log', 'a') as f:
    #     f.write('%s\n\n@\n\n' % y_hat)
    # 计算正确率，不算也行，但是早停需要用所以就进行计算
    accuracy = tf.math.reduce_mean((y_hat*y)+((1-y_hat)*(1-y)))
    test(self)
    return {'loss': loss, 'accuracy': accuracy}


for root, dirs, files in os.walk('./imgs', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

model = net.network(NETWORK)

model.compile(
    optimizer=opts.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy']
)
X, Y = init_data(1000)
model.train_step = functools.partial(
    step, model
)  # partial负责把model赋给step的self参数
model.summary()
model.fit(
    tf.convert_to_tensor(X, dtype='float32'),
    tf.convert_to_tensor(Y, dtype='float32'),
    epochs=10
)

# a, b = init_data(1)
