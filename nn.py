import os
import sys

import numpy as np
import tensorflow as tf

TRAIN = 0.8
LOSS = 0

variable_cnt = 0

def _get_variable(name, shape, wd=.0):
  global LOSS, variable_cnt
  stddev = 1.0
  tmp = 1
  for t in shape:
    stddev /= t
    tmp *= t
  if tf.get_variable_scope().reuse != True:
    variable_cnt += tmp
  stddev = stddev ** 0.5
  ret = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(dtype=tf.float32, mean=0.0, stddev=stddev))

# for tensorboard
  tf.summary.tensor_summary(name, ret)
  tf.summary.scalar(name + 'mean', tf.reduce_mean(ret))

  if wd != .0:
    LOSS += wd * tf.nn.l2_loss(ret)
  return ret

def fc(x, o_shape=None, name='fc'):
  with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    if not o_shape:
      o_shape = shape[1:]
    batch = shape[0]

    size = 1
    for i in shape[1:]:
      size *= i
    o_size = 1
    for i in o_shape:
      o_size *= i

    k = _get_variable('weights', [size, o_size])
    b = _get_variable('biases', [o_size])

    x = tf.reshape(x, [batch, size])
    x = tf.matmul(x, k) + b
    x = tf.reshape(x, [batch] + o_shape)
    return tf.nn.relu(x)

def afc(x, o_shape=None, name='afc'):
  with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    if not o_shape:
      o_shape = shape[1:]
    trans_perm = [0] + [i for i in range(2, len(shape))] + [1]
    for i in range(1, len(shape)):
      sshape = x.get_shape().as_list()
      x = tf.expand_dims(x, -2)
      now_size = o_shape[-2 + i]
      w = _get_variable('weights_' + str(i), sshape + [now_size])
      b = _get_variable('biases_' + str(i), sshape[:-1] + [now_size])
      x = tf.reshape(tf.matmul(x, w), sshape[:-1] + [now_size]) + b
      x = tf.transpose(x, trans_perm)
    return tf.nn.relu(x)


def pool(x, name='pool'):
  with tf.variable_scope(name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def unpool(x, name='unpool', shape=None):
  with tf.variable_scope(name):
    if not shape:
      shape = x.get_shape().as_list()
      shape = [2*shape[1], 2*shape[2]]
    else:
      shape = [shape[1], shape[2]]
    return tf.image.resize_nearest_neighbor(x, shape, name=name)

def conv(x, o_size, name='conv', ksize=3):
  with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    k = _get_variable('weights', [ksize, ksize, shape[3], o_size])
    b = _get_variable('biases', [1, 1, 1, o_size])
    return tf.nn.relu(tf.nn.conv2d(x, k, [1, 1, 1 ,1], 'SAME') + b)

def deconv(x, o_shape, name='deconv', ksize=4, stride=2):
  pass

def dropout(x, name='dropout'):
  with tf.variable_scope(name):
    if TRAIN:
      return tf.nn.dropout(x, TRAIN)
    else:
      return x

def layer_add(x, y, name='layer_add'):
  with tf.variable_scope(name):
    wx = _get_variable('weight_x', [1])
    wy = _get_variable('weight_y', [1])
    b = _get_variable('biases', [1])
    return tf.nn.relu(wx * x + wy * y + b)
