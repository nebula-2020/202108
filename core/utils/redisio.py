#!~/.conda/envs python
# -*- coding: utf-8 -*-
import base64
import struct

import numpy
import redis

HOST = 'localhost'
PORT = 6379
connection_pool = None


def connect():
    global connection_pool
    connection_pool = redis.ConnectionPool(
        host=HOST,
        port=PORT,
        decode_responses=True
    )  # 连接池


def redis_save(key: str, numpy_ndarray: numpy.ndarray) -> None:
    """将Numpy数组存入Redis数据库。

    Parameters
    ----------
    key : str
        键字符串。
    numpy_ndarray : numpy.ndarray
        待存储数组。
    """
    shape = numpy_ndarray.shape
    dim = len(shape)
    value = struct.pack(''.join(['>I']+['I'*dim]), *((dim,)+shape))
    value = base64.a85encode(value+numpy_ndarray.tobytes())
    conn = redis.Redis(connection_pool=connection_pool)
    conn.set(key, value)
    conn.close()


def redis_read(key: str, dtype) -> numpy.ndarray:
    """从Redis中读取一个Numpy数组。

    Parameters
    ----------
    key : str
        键字符串。
    dtype : Any
        指定数组元素数据类型。
    Returns
    -------
    numpy.ndarray
        从Redis键值对取出的数组。
    """
    SIZE = 4
    conn = redis.Redis(connection_pool=connection_pool)
    bytes = base64.a85decode(conn.get(key))
    conn.close()
    dim = struct.unpack('>I', bytes[:1*SIZE])[0]
    shape = struct.unpack('>%s' % ('I'*dim), bytes[1*SIZE:(dim+1)*SIZE])
    ret = numpy.frombuffer(
        bytes,
        offset=(dim+1)*SIZE,
        dtype=dtype
    ).reshape(shape)
    return ret


def redis_close():
    connection_pool.disconnect()
