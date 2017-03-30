import os
import sys
import random

import numpy as np
import tensorflow as tf

import model
import test

first_day = 16884
last_day = 17249

train_rate = 1e-5

BATCH = 1
LENGTH = 12
I_SHAPE = [20, 19]
O_SHAPE = [20, 2]

_x = tf.placeholder(shape=[BATCH] + [LENGTH] + I_SHAPE, dtype=np.float32)
_y = tf.placeholder(shape=[BATCH] + [LENGTH] + O_SHAPE, dtype=np.float32)

_x_list = tf.split(_x, _x.get_shape().as_list()[1], axis=1)

# MEMARY CUBE

_mem_0 = tf.placeholder(shape=[BATCH] + [20,30,30], dtype=np.float32)
_mem_1 = tf.placeholder(shape=[BATCH] + [20,40,40], dtype=np.float32)

mem = [_mem_0, _mem_1]

o = []

mem, tmp = model.rnn_cell(mem, _x_list[0])
o.append(tmp)

for i in _x_list[1:]:
  mem, tmp = model.rnn_cell(mem, i, reuse=True)
  o.append(tmp)

o = tf.stack(o, axis=1)

Z = np.zeros([BATCH, 1, 20, 2])

loss = tf.nn.l2_loss(tf.concat([Z, o], axis=1) - tf.concat([_y, Z], axis=1))

opt = tf.train.AdamOptimizer(train_rate).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(0, 1000):
  print('EPOCH %d' % i)

  mem_0 = np.zeros(shape=[BATCH] + [20,30,30])
  mem_1 = np.zeros(shape=[BATCH] + [20,40,40])

  for date in range(first_day + random.randint(0, LENGTH), last_day - LENGTH, LENGTH):
    x = test.foo6(date, LENGTH)
    y = np.split(x, 19, axis=2)
    y = np.concatenate([y[7], y[8]], axis=2)
    x = np.stack([x])
    y = np.stack([y])

    sess.run([opt], feed_dict={_x: x, _y: y, _mem_0: mem_0, _mem_1: mem_1})
    mem_0, mem_1, lo = sess.run([_mem_0, _mem_1, loss], feed_dict={_x: x, _y: y, _mem_0: mem_0, _mem_1: mem_1})
    print('      loss = %.2e' % lo)