import os
import sys

import numpy as np
import tensorflow as tf

import nn

I_SHAPE = [20, 19]
O_SHAPE = [20, 2]

def rnn_cell(mem, x, name='rnn_cell', reuse=False):
  with tf.variable_scope(name, reuse=reuse):

    with tf.variable_scope('thinging_1'):
      for i in range(0, len(mem)):
        mem[i] = nn.afc(mem[i], name='afc_' + str(i))

    with tf.variable_scope('recall'):
      for i in range(0, len(mem) - 1):
        mem[i] = nn.layer_add(nn.afc(mem[i + 1], mem[i].get_shape().as_list()[1:], name='afc_' + str(i)), mem[i], name='layer_add_' + str(i))

    with tf.variable_scope('thinging_2'):
      for i in range(0, len(mem)):
        mem[i] = nn.afc(mem[i], name='afc_' + str(i))

    with tf.variable_scope('in_flow'):
      x = nn.afc(x, name='afc_1')
      x = nn.afc(x, name='afc_2')
      x = nn.afc(x, name='afc_3')
      mem[0] = nn.layer_add(nn.fc(x, mem[0].get_shape().as_list()[1:]), mem[0])

    with tf.variable_scope('out_flow'):
      x = nn.layer_add(nn.fc(mem[0], O_SHAPE, name='fc_mem'), nn.fc(x, O_SHAPE, name='fc_x'))
      x = nn.afc(x, name='afc_1')
      x = nn.afc(x, name='afc_2')
      x = nn.afc(x, name='afc_3')

    with tf.variable_scope('thinging_3'):
      for i in range(0, len(mem)):
        mem[i] = nn.afc(mem[i], name='afc_' + str(i))

    with tf.variable_scope('remember'):
      for i in range(len(mem) - 1, 0, -1):
        mem[i] = nn.layer_add(nn.afc(mem[i - 1], mem[i].get_shape().as_list()[1:], name='afc_' + str(i)), mem[i], name='layer_add_' + str(i))

    with tf.variable_scope('thinging_4'):
      for i in range(0, len(mem)):
        mem[i] = nn.afc(mem[i], name='afc_' + str(i))

    return [mem, x]
