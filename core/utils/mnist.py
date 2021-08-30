#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取MNIST数据集。
"""
import gzip
import os
import struct

from numpy import reshape
from numpy.random import randint, randn

TRAIN_X_FILENAME = 'train-images-idx3-ubyte.gz'  # 训练集图片
TRAIN_Y_FILENAME = 'train-labels-idx1-ubyte.gz'  # 训练集标签
TEST_X_FILENAME = 't10k-images-idx3-ubyte.gz'  # 测试集图片
TEST_Y_FILENAME = 't10k-labels-idx1-ubyte.gz'  # 测试集标签


def read_mnist(root: str, noise=False):
    """读取整个MNIST数据集

    Args:
        root (str): MNIST数据集的`train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`、`t10k-images-idx3-ubyte.gz`、`t10k-labels-idx1-ubyte.gz`四个`.gz`文件所在目录。
        noise (bool, optional): 描述是否添加噪声. 默认`False`。

    Returns:
        tuple: 训练集图片、训练集标签、测试集图片、测试集标签、图片尺寸。
    """

    STEP = 4  # 4个字节一起读取
    MODE = '>'  # 大端模式读

    def read(x_name, y_name, root, noise):
        with gzip.open(os.path.join(root, x_name), 'rb') as gz:
            data = gz.read()
        count, width, height = struct.unpack(
            '%sIII' % MODE,
            data[1*STEP:4*STEP]
        )
        shape = (height, width, 1)
        x = list(struct.unpack_from(
            '%s%dB' % (MODE, width*height*count), data, 4*STEP
        ))
        if noise:
            for i in range(len(x)):
                r = randint(-1, 2)
                if r == 0:
                    x[i] = x[i]
                elif r == 1:
                    x[i] = min(max(x[i] * randn(), 0), 1)
                else:
                    x[i] = min(max(x[i] + r * randn(), 0), 1)
        x = reshape(x, (count, height, width, 1))/255.
        with gzip.open(os.path.join(root, y_name), 'rb') as gz:
            data = gz.read()
        y = [
            [1 if i_ == int(i) else 0 for i_ in range(10)] for i in data
        ][2*STEP:]
        return x, y, shape
    x, y, s = read(TRAIN_X_FILENAME, TRAIN_Y_FILENAME, root, noise)
    x_t, y_t, _ = read(TEST_X_FILENAME, TEST_Y_FILENAME, root, noise)
    return x, y, x_t, y_t, s
