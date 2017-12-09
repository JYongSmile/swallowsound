# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# from swallowsoundml import input_data
from swallowsound.swallowsound_input_data import read_data_sets

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    # 注意one_hot参数表示我们要吧标签生成向量，比如目前的标签是数字“7”，分类数为10，则要生成一个[0,0,0,0,0,0,1,0,0,0]的向量
    # 返回的结果是一个数据集，主要包括['train', 'validation', 'test']三个数据集，其中train和test的大小由原始数据确定，validation
    # 从原始数据的train集合中选取，其大小要小于train的大小，具体大小由validation_size参数确定
    # 数据集合中样本的分类数目由num_classes参数设置，标签label的值必须是[0，num_classes)之间的整数
    num_classes = 2
    # swallowsound = read_data_sets(FLAGS.data_dir,
    #                             gzip_compress=False,
    #                             train_imgaes='LearnSamples.bin',
    #                             train_labels='LearnSamplesflag.bin',
    #                             test_imgaes='TestSamples.bin',
    #                             test_labels='TestSamplesflag.bin',
    #                             one_hot=True,
    #                             validation_size=50,
    #                             num_classes = num_classes,
    #                             MSB=False)

    swallowsound = read_data_sets(FLAGS.data_dir,
                                gzip_compress=False,
                                train_imgaes='train-images-idx3-ubyte',
                                train_labels='train-labels-idx1-ubyte',
                                test_imgaes='t10k-images-idx3-ubyte',
                                test_labels='t10k-labels-idx1-ubyte',
                                one_hot=True,
                                validation_size=50,
                                num_classes=num_classes,
                                MSB=True)

    original_shape = swallowsound.train.original_shape
    if len(original_shape)<4:
      return

    size = original_shape[1]*original_shape[2]

    # Create the model
    x = tf.placeholder(tf.float32, [None, size])
    W = tf.Variable(tf.zeros([size, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = swallowsound.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: swallowsound.test.images,
                                      y_: swallowsound.test.labels}))

    print("b: ",sess.run(b))
    # print("W: ", sess.run(W))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/noise/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
