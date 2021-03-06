#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/8 15:26
# @Author  : Barry_J
# @Email   : s.barry1994@foxmail.com
# @File    : noise_deep_save.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from swallowsound.swallowsound_input_data import read_data_sets

import tensorflow as tf

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 1, 50, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([1, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_1x5(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([1, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_1x5(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([1 * 2 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 1*2*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x5(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                        strides=[1, 1, 5, 1], padding='SAME')

#def max_pool_1x5(x):
  #"""max_pool_5x5 downsamples a feature map by 2X."""
  #return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                        #strides=[1, 1, 5, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  num_classes = 2
  swallowsound = read_data_sets(FLAGS.data_dir,
                                gzip_compress=False,
                                train_imgaes='train-images-idx3-ubyte',
                                train_labels='train-labels-idx1-ubyte',
                                test_imgaes='t10k-images-idx3-ubyte',
                                test_labels='t10k-labels-idx1-ubyte',
                                one_hot=True,
                                validation_size=2000,#验证集大小
                                num_classes=num_classes,
                                MSB=True)



  # Create the model
  x = tf.placeholder(tf.float32, [None, 50])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#accuracy 0.97
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)         #accuracy 0.94


  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
      batch = swallowsound.train.next_batch(500)#batch 200 比 50的效果好  # batch(500) test accuracy 0.985714
      if i % 500 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: swallowsound.test.images, y_: swallowsound.test.labels, keep_prob: 1.0}))



# def save(self,
#          sess,
#          save_path):
#
#     saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     #训练模型过程
#     #使用saver提供的简便方法去调用 save op
#     saver.save(sess, "save_path/file_name.ckpt")


# 声明两个变量







if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/noise/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




# v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
# v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
# init_op = tf.global_variables_initializer() # 初始化全部变量
# saver = tf.train.Saver(write_version=tf.train.SaverDef.V1) # 声明tf.train.Saver类用于保存模型
# with tf.Session() as sess:
#     sess.run(init_op)
#     print("v1:", sess.run(v1)) # 打印v1、v2的值一会读取之后对比
#     print("v2:", sess.run(v2))
#     saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
#     print("Model saved in file:", saver_path)