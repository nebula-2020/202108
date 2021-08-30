#!~/.conda/envs python
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.layers as layers
import core.utils.network as net


def yolo_network(IMG_SHAPE: tuple, TAR_SHAPE: tuple) -> list:
    return [
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
        layers.Dense(units=800, activation='tanh'),
        layers.Reshape(target_shape=TAR_SHAPE),
    ]
