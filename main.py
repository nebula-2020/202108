#!~/.conda/envs python
# -*- coding: utf-8 -*-
from core.yolo_structure import yolo_network
from core.utils.captchagenerator import create, show
import tensorflow as tf
import functools  # 来源Python标准库的包，用于处理函数和可调用对象的工具
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as opts
import core.utils.network as net
import numpy as np
import PIL.Image as Image
from tensorflow.python.keras.engine import data_adapter
import time
from tensorflow.python.keras.callbacks import Callback  # 每轮拟合都调用一次的回调函数类的基类
import os
tf.config.experimental_run_functions_eagerly(True)
np.set_printoptions(suppress=True, threshold=np.inf)  # 设置不用科学计数法

BOXES = [0.8, 1., 1.25]
BOXES.sort()
BOX_COUNT = len(BOXES)
CLASS_COUNT = 10
IMG_SHAPE = (56, 112, 1)
TAR_SHAPE = (4, 8, BOX_COUNT*5+CLASS_COUNT)
EXPORT_PATH_IMG = './export/imgs'
EXPORT_PATH_TXT = './export/txts'
NETWORK = yolo_network(IMG_SHAPE, TAR_SHAPE)
NETWORK_SAVE_PATH = './export/model/yolo'
LAMBDA_COORD = 5.
LAMBDA_NOOBJ = .5


def init_data_single(labels):  # 生成单一数据
    cell_h = IMG_SHAPE[0] / TAR_SHAPE[0]  # Cell的高度
    cell_w = IMG_SHAPE[1] / TAR_SHAPE[1]  # Cell的宽度
    labels_ = []
    for label in labels:  # 遍历所有物体
        x, y, w, h, class_ = label  # 解包label元组，x、y、w、h全部以像素为单位
        y_index = min(TAR_SHAPE[0]-1, int((y+h/2)/cell_h))  # Cell列标
        x_index = min(TAR_SHAPE[1]-1, int((x+w/2)/cell_w))  # Cell行标

        # y坐标加上一半高度算出中心y坐标，减去上面Cell总高度，算出距离当前Cell上边多少像素，除以Cell高度得到中心距离当前Cell上边百分之多少
        y = (y+h/2-cell_h*y_index)/cell_h

        # x坐标加上一半宽度算出中心x坐标，减去左面Cell总宽度，算出距离当前Cell左边多少像素，除以Cell宽度得到中心距离当前Cell左边百分之多少
        x = (x+w/2-cell_w*x_index)/cell_w

        w, h = w/IMG_SHAPE[1], h/IMG_SHAPE[0]  # 求相对于图像的宽度和高度

        for bi in range(len(BOXES)):
            if w/h <= BOXES[bi]:
                box_index = bi
                if BOXES[bi] < 1:
                    w *= BOXES[bi]
                else:
                    h /= BOXES[bi]
                break
        labels_.append((x_index, y_index, box_index, x, y, w, h, class_))  # 打包
    # 真实值，但是这里面包含的很多没用的nan,这些nan单纯用来占位，什么都不做
    targets = np.full(TAR_SHAPE, np.nan, dtype=float)

    for yi_ in range(TAR_SHAPE[0]):  # 遍历行
        for xi_ in range(TAR_SHAPE[1]):  # 遍历列
            for bi_ in range(BOX_COUNT):  # 遍历Box
                targets[yi_, xi_, bi_] = 0  # 绝大多数box置信度都是0
    for label in labels_:  # 遍历所有物体
        xi, yi, bi, x, y, w, h, class_ = label  # 解包label元组，全部相对位置相对尺寸
        vals = (x, y, w, h)  # 打包省代码
        targets[yi, xi, bi] = 1.  # 设置置信度
        # 设置box尺寸真实值
        for i in range(4):  # 遍历x,y,w,h五个值
            targets[yi, xi, BOX_COUNT + bi*4+i] = vals[i]  # 设置每一个真实值
        # 设置类别
        for ci in range(CLASS_COUNT):  # 遍历类别
            if ci == class_:  # 是物体真实类别
                targets[yi, xi, BOX_COUNT*5+ci] = 1.  # 正确类别的真实值是1
            else:  # 不是物体真实类别
                targets[yi, xi, BOX_COUNT*5+ci] = 0.  # 错误类别应该是0
    # print(labels_)
    return targets


def init_data(size: int):
    imgs, targets = [], []
    for _ in range(size):
        img, labels = create()  # 创建验证码，img没有颜色深度，labels见下文
        target = init_data_single(labels)
        imgs.append(img)
        targets.append(target)
    return np.array(imgs), np.array(targets)


def test(model):
    cell_h = IMG_SHAPE[0] / TAR_SHAPE[0]  # Cell的高度
    cell_w = IMG_SHAPE[1] / TAR_SHAPE[1]  # Cell的宽度
    image, ls = create()  # 创建验证码，img没有颜色深度，labels见下文
    ls = ['%d' % e[-1] for e in ls]
    img = np.reshape(image, IMG_SHAPE)
    y_hat = model(tf.convert_to_tensor([img]), training=False)
    y_hat = y_hat[0]
    labels = []
    for xi in range(TAR_SHAPE[1]):  # 遍历列
        for yi in range(TAR_SHAPE[0]):  # 遍历行
            for bi in range(BOX_COUNT):  # 遍历Box
                if y_hat[yi, xi, bi] >= .5:  # 负责预测的box
                    # # 计算中心相对x坐标
                    x = (xi+y_hat[yi, xi, BOX_COUNT + bi*4+0])*cell_w
                    x = x-y_hat[yi, xi, BOX_COUNT + bi*4+2]/2*IMG_SHAPE[1]

                    # 计算中心相对y坐标
                    y = (yi+y_hat[yi, xi, BOX_COUNT + bi*4+1])*cell_h
                    y = y-y_hat[yi, xi, BOX_COUNT + bi*4+3]/2*IMG_SHAPE[0]

                    # 计算宽、高
                    w = y_hat[yi, xi, BOX_COUNT + bi*4+2]*IMG_SHAPE[1]
                    h = y_hat[yi, xi, BOX_COUNT + bi*4+3]*IMG_SHAPE[0]
                    if w/h <= BOXES[bi]:
                        if BOXES[bi] < 1:
                            w /= BOXES[bi]
                        else:
                            h *= BOXES[bi]
                    labels.append(
                        (
                            x, y, w, h,
                            np.argmax(y_hat[yi, xi, -CLASS_COUNT:])  # class
                        )
                    )
    image_, title = show(image, labels)
    name = int(time.time()*1e7)
    with open('%s/%d.txt' % (EXPORT_PATH_TXT, name), 'w') as f:
        f.write(str(y_hat))
    image_.save(
        '%s/%d_%s_%s.png' % (
            EXPORT_PATH_IMG,
            name,
            title,
            ''.join(ls)
        ), 'png'
    )


@tf.function  # 这个注解控制将方法加入内存以提高运行速度
def step(self, data):  # 定义训练步骤方法，模型每轮拟合都进行
    x, y_raw, _ = data_adapter.unpack_x_y_sample_weight(data)  # 把元组拆成单个值
    y_h = self(x, training=False)  # 非训练进行一次计算，得到模型当前参数输出
    y = tf.where(tf.math.is_nan(y_raw), x=y_h, y=y_raw)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(self.trainable_variables)  # 监控网络参数进行过的运算
        y_hat = self(x, training=True)  # 得出模型输出
        loss = self.loss(y, y_hat)  # 计算模型损失
    grads = tape.gradient(loss, self.trainable_variables)  # GradientTape 自动求导
    self.optimizer.apply_gradients(  # 优化网络参数
        zip(grads, self.trainable_variables)  # 打包为元组
    )
    return {'loss': loss}


class Callback_(Callback):  # 定义回调类
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def on_epoch_end(self, epoch, logs={}):  # self就是Model对象的指针
        test(self.model)


def recreate_exportdirs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if not os.path.exists(path):
        os.mkdir(path)


recreate_exportdirs(EXPORT_PATH_IMG)
recreate_exportdirs(EXPORT_PATH_TXT)
# model = keras.models.load_model(NETWORK_SAVE_PATH)
model = net.network(NETWORK)

model.compile(
    optimizer=opts.Adam(learning_rate=0.0002),
    loss=tf.keras.losses.MeanSquaredError(),
)
X, Y = init_data(5000)
model.train_step = functools.partial(
    step, model
)  # partial负责把model赋给step的self参数
# model.summary()
test(model)
model.fit(
    tf.convert_to_tensor(X, dtype='float32'),
    tf.convert_to_tensor(Y, dtype='float32'),
    epochs=500,
    callbacks=[Callback_(model)]
)
model.save(NETWORK_SAVE_PATH)
