# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)
# 读入mnist数据集
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# 变量声明
# define placeholder for inputs to network
with tf.name_scope('inputs'):
    x = tf.placeholder('float', [None, 784], name='x_input')  # None 为了留给后面的batch_size
    y_ = tf.placeholder('float', [None, 10], name='y_input')

with tf.name_scope('conv1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='W1')
        tf.summary.histogram('conv1/weights', W_conv1)
    with tf.name_scope('bias'):
        b_conv1 = bias_variable([32], name='B1')
        tf.summary.histogram('conv1/bias', b_conv1)
with tf.name_scope('conv2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W2')
        tf.summary.histogram('conv2/weights', W_conv2)
    with tf.name_scope('bias'):
        b_conv2 = bias_variable([64], name='B2')
        tf.summary.histogram('conv2/bias', b_conv2)
with tf.name_scope('conv3'):
    with tf.name_scope('weights'):
        W_conv3 = weight_variable([5, 5, 64, 4], name='W3')
        tf.summary.histogram('conv2/weights', W_conv3)
    with tf.name_scope('bias'):
        b_conv3 = bias_variable([4], name='B3')
        tf.summary.histogram('conv3/bias', b_conv3)

with tf.name_scope('fc1'):
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([7 * 7 * 4, 1024], name='w1')
        tf.summary.histogram('fc1/weights', W_fc1)
    with tf.name_scope('bias'):
        b_fc1 = bias_variable([1024], name='b1')
        tf.summary.histogram('fc1/weights', b_fc1)
with tf.name_scope('fc2'):
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([1024, 10], name='w2')
        tf.summary.histogram('fc2/weights', W_fc2)
    with tf.name_scope('bias'):
        b_fc2 = bias_variable([10], name='b2')
        tf.summary.histogram('fc2/weights', b_fc2)
# 计算图
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 10)

with tf.name_scope('c1'):
    conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('c2'):
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('c3'):
    conv3 = tf.nn.relu(tf.nn.conv2d(pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    tf.summary.image('third_conv', conv3, 10)

with tf.name_scope('f1'):
    pool2_flat = tf.reshape(conv3, [-1, 7*7*4])
    fc1 = tf.nn.relu(tf.matmul(pool2_flat,W_fc1) + b_fc1)

with tf.name_scope('f2'):
    y = tf.nn.softmax(tf.matmul(fc1, W_fc2) + b_fc2)

with tf.name_scope('loss'):
    loss = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    tf.summary.scalar('accuracy', accuracy)
# 初始化变量
init = tf.global_variables_initializer()
#logs
merged_summary_op = tf.summary.merge_all()

# 开始训练
with tf.Session() as sess:
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    test_summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    for i in range(100):
        x_batch, y_batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})
        if i % 10 == 0:
            print(sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch}))
            #eval on train set
            summary_str = sess.run(merged_summary_op, feed_dict={x: x_batch, y_: y_batch})
            train_summary_writer.add_summary(summary_str, i)
            #eval on test set
            summary_str = sess.run(merged_summary_op, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            test_summary_writer.add_summary(summary_str, i)

    print ("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))
