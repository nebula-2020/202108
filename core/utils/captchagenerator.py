#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证码生成和展示。
"""
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
randint = np.random.randint

font_HOME = '/usr/share/fonts/truetype/ubuntu'  # 字体路径


def create(
    size: tuple = (56, 112),  # 验证码图片尺寸
    font: str = 'Ubuntu-L.ttf',  # 字体文件名
    font_size: int = 32,  # 字号
    word_count: int = None,  # 数字数量
    line_count: int = 64,  # 斑点数量
    line_len: int = 6,  line_width: int = 3,  # 斑点尺寸
    noise: bool = True
) -> tuple:
    """创建验证码。

    Parameters
    ----------
    size : tuple, optional
        验证码高宽(y,x), by default (56, 112)
    font : str, optional
        字体名称, by default 'Ubuntu-L.ttf'
    font_size : int, optional
        字体大小, by default 32
    word_count : int, optional
        验证码位数, by default 4
    line_count : int, optional
        杂线数量, by default 64
    line_len : int, optional
        杂线长度, by default 6
    line_width : int, optional
        杂线宽度, by default 3
    noise : bool, optional
        噪声开关, by default True
    Returns
    -------
    tuple
        numpy.ndarray, 五元组。
        其中五元组为(left,top,width,height,num)，坐标以像素为单位。
    """
    if word_count is None:
        word_count = random.randint(0, 6)
    if noise and random.randint(0, 2) == 0:
        img = np.uint8(np.uint8((np.random.randn(*size)+2)*255)*0.75)  # 新建噪声
    else:
        img = np.zeros(size, dtype='uint8')
    img = Image.fromarray(img)  # 对象类型转换
    draw = ImageDraw.Draw(img)  # 得到绘图对象
    if noise and random.randint(0, 2) == 0:
        for _ in range(line_count):  # 绘制斑点
            y, x = randint(0, size[0]), randint(0, size[1])  # 随机化斑点位置
            r = randint(1, line_len)  # 随机斑点半径
            draw.line(
                (
                    x-randint(-r, r),  y-randint(-r, r),  # 起始坐标
                    x+randint(-r, r),   y+randint(-r, r)  # 终止坐标
                ),  # 必须打包为元组
                '#ffffff',  # 颜色值，因为是灰度图像，所以除了黑白灰之外的颜色都只能转换成黑白灰
                randint(1, line_width)  # 随机宽度最小为1
            )  # 画出斑点
    txt = [randint(0, 10) for _ in range(word_count)]  # 随机化10个数字
    labels = []  # 样本标签初始化

    simsun = ImageFont.truetype(
        os.path.join(font_HOME, font),  # 路径
        font_size-randint(0, 1+int(font_size/5))  # 字大小
    )  # 字体对象
    for i in range(word_count):  # 每一个数字都有标签
        t_layer = Image.new("RGBA", (font_size,)*2, (0, 0, 0, 0))  # 再新建一个图层，写字
        t_draw = ImageDraw.Draw(t_layer)  # 绘图对象
        tw, th = t_draw.textsize(str(txt[i]), font=simsun)  # 获取字大小
        # 为什么不直接把图层大小设置成字大小呢，因为字绘制时有行高会留出空白，这样字就超出图层
        t_draw.text(
            xy=(int((font_size-tw)/2), int((font_size-th)/2)),  # 文字左上角位置
            text=str(txt[i]),  # 文本
            font=simsun,  # 字体
            fill='#ffffff'  # 颜色值为白色
        )  # 画一个字出来
        t_layer = t_layer.resize(
            (
                int(font_size*(.75+random.rand()*.25)),
                int(font_size*(.75+random.rand()*.25))
            )
        )  # 进行拉伸变形
        t_layer = t_layer.rotate(randint(-30, 31))  # 略微旋转

        arr = np.array(t_layer)  # 类型转换

        # 下面查找边界框
        sign = False  # 描述当前是否已经检测到一个边界
        left, right = 0, arr.shape[-1]-1  # 先检测左右边界
        row = np.max(arr, axis=0)  # 纵列求和得一行
        for c in range(len(row)):
            if row[c][-1] > 0 and not sign:  # 某列像素不是全黑的说明这个像素是数字的一部分
                left = c  # 检测到了一个边界
                sign = True  # 设置记号
            elif row[c][-1] <= 0 and sign:  # 有一列像素全黑，说明已经超出数字部分
                right = c-1  # 检测到另一个边界
                break  # 停止遍历

        # 同理……
        sign = False
        top, bottom = 0, arr.shape[0]-1
        col = np.max(arr, axis=1)
        for r in range(len(col)):
            if col[r][-1] > 0 and not sign:
                top = r
                sign = True
            elif col[r][-1] <= 0 and sign:
                bottom = r-1
                break

        t_layer = Image.fromarray(arr[top:bottom, left:right])  # 裁剪图层

        LAYER_size = t_layer.size  # 获得图层尺寸

        # 合并图层
        offset_y = int(size[0]/word_count/2)  # 随机偏移值
        offset_x = int(size[1]/word_count/4)  # 随机偏移值
        x = max(
            0,
            min(
                size[1]-LAYER_size[1],
                (i+.5)*size[1] / word_count
                - LAYER_size[1] / 2
                + randint(- offset_x, offset_x + 1)
            )
        )  # 图层左边坐标
        y = max(
            0,
            min(
                size[0]-LAYER_size[0],
                size[0]/2-LAYER_size[0]/2 + randint(-offset_y, offset_y+1)
            )
        )  # 图层顶部坐标
        draw.bitmap(
            (x, y),
            t_layer,
            fill='#ffffff'
        )  # 合并图层，将数字所在的图层合并到噪声图层
        labels.append(
            [
                x,  # 边界框左坐标
                y,  # 边界框上坐标
                (right-left),  # 边界框宽度
                (bottom-top),  # 边界框高度
                txt[i]  # 数字值
            ]
        )
    img = np.array(img)
    if noise and random.randint(0, 2) == 0:
        img = 255-img
    return img, labels


def show(image: np.ndarray, labels: list) -> tuple:
    """创建图片对象。

    Parameters
    ----------
    image : np.ndarray
        二维矩阵
    labels : list
        五元组为(left,top,width,height,num)，坐标以像素为单位

    Returns
    -------
    tuple
        (PIL.Image, str)
    """
    img = Image.fromarray(image)  # 对象类型转换
    img = img.convert('RGB')  # 转换成RGB，为了画彩色边界框
    draw = ImageDraw.Draw(img)  # 准备绘制边界框
    title = ''  # 标题初始化
    for label in labels:
        draw.rectangle(
            (
                int(label[0]),
                int(label[1]),
                int(label[2]+label[0]),
                int(label[3]+label[1])
            ),
            fill=None, outline='#ff0000'
        )  # 画一个框
        title += str(label[4])  # 然后加一个字
    return img, title


if __name__ == '__main__':
    img, labels = create()
    img, title = show(img, labels)
    plt.imshow(img)
    plt.title(title)
    plt.show()
