import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import cv2
import cuhk03_dataset

import os
import sys
import shutil
import math
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160

def preprocess(images, is_train):
    def train():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.random_flip_left_right(split[i][j])
                split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    def val():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    return tf.cond(is_train, train, val)

def network(images1, images2, weight_decay):
    with tf.variable_scope('network'):
        image_shaped_input = tf.reshape(images1, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
#        tf.summary.image('input1', images1)
        tf.summary.image('input1', image_shaped_input)
        tf.summary.image('input2', images2)
        # Tied Convolution
        conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_1')
        conv1_1_transpose=tf.reshape(conv1_1, [1, 140, 312, 4])
#        conv1_1_transpose=tf.transpose(conv1_1, [3, 0, 1, 2])
 #       fig1,ax1=plt.subplots(nrows=1, ncols=20, figsize = (20,1))
  #      for i in range(20):
   #         ax1[i].imshow(conv1_1_transpose[i][0])
    #    plt.title('Conv1 20*156*56')
     #   plt.imsave('./test1.jpg')
#        concat1=[]
 #       for i in range(5):
  #          concat1.append(conv1_1_transpose[i])
   #     tf.summary.image("conv1_1", concat1)
#        conv1_1_transpose=tf.transpose(conv1_1_transpose, [0, 3, 1, 2])
 #       concat1=conv1_1_transpose[0]
  #      print("concat1: "+str(concat1))
        #for i in range(1,5):
         #   for j in range(4):
          #      print ("concat1[j] : "+str(concat1[j]))
           #     print ("conv1_1_transpose[i][j] : "+str(conv1_1_transpose[i][j]))
            #    concat1[j]=tf.add(concat1[j],conv1_1_transpose[i][j])
#        concat1_transpose=tf.transpose(concat1, [1, 2, 0])
        #concat1_transpose=[1,concat1_transpose]
 #       concat1_transpose=tf.reshape(concat1, [-1, 156, 56, 4])
  #      tf.summary.image("conv1_1", concat1_transpose)
        #tf.summary.image("conv1_1", conv1_1_transpose)
#        tf.summary.image("conv1_1", conv1_1)
        tf.summary.histogram('conv1_1', conv1_1)
        pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
#        pool1_1_transpose=tf.reshape(pool1_1, [1, 140, 104, 3])
#        pool1_1_transpose=tf.reshape(pool1_1, [1, 140, 78, 4])
        pool1_1_transpose=tf.reshape(pool1_1, [-1, 78, 28, 4])
        print("images1: "+str(images1)+"   pool1_1 : "+str(pool1_1))
        pool1_1_transpose=tf.transpose(pool1_1, [3, 1, 2, 0])
#        tf.summary.image("pool1_1", tf.reshape(pool1_1_transpose[19],[1,78,28,-1]))
        tf.summary.image("pool1_1", pool1_1_transpose,1)
#        tf.summary.image("pool1_1", pool1_1_transpose,10)
 #       tf.summary.image("pool1_1", pool1_1)
        tf.summary.histogram('pool1_1', pool1_1)
        conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_2')
  #      tf.summary.image("conv1_2", conv1_2)
        conv1_2_transpose=tf.reshape(conv1_2, [1, 200, 74, 3])
       # tf.summary.image("conv1_2", conv1_2_transpose)
        tf.summary.histogram('conv1_2', conv1_2)
        pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')
   #     tf.summary.image("pool1_2", pool1_2)
        pool1_2_transpose=tf.reshape(pool1_2, [1, 100, 37, 3])
        #tf.summary.image("pool1_2", pool1_2_transpose)
        pool1_2_transpose=tf.transpose(pool1_2, [3, 1, 2, 0])
        tf.summary.image("pool1_2", pool1_2_transpose,1)
        tf.summary.histogram('pool1_2', pool1_2)
        conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_1')
    #    tf.summary.image("conv2_1", conv2_1)
        #conv2_1_transpose=tf.reshape(conv2_1, [-1, 156, 56, 4])
        #tf.summary.image("conv2_1", conv2_1_transpose,5)
        conv2_1_transpose=tf.reshape(conv2_1, [1, 140, 312, 4])
        #tf.summary.image("conv2_1", conv2_1_transpose)
        tf.summary.histogram('conv2_1', conv2_1)
        pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
     #   tf.summary.image("pool2_1", pool2_1)
        pool2_1_transpose=tf.reshape(pool2_1, [1, 140, 78, 4])
        #tf.summary.image("pool2_1", pool2_1_transpose)
        pool2_1_transpose=tf.transpose(pool2_1, [3, 1, 2, 0])
        tf.summary.image("pool2_1", pool2_1_transpose,1)
        tf.summary.histogram('pool2_1', pool2_1)
        conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_2')
      #  tf.summary.image("conv2_2", conv2_2)
        conv2_2_transpose=tf.reshape(conv2_2, [1, 200, 74, 3])
        #tf.summary.image("conv2_2", conv2_2_transpose)
        tf.summary.histogram('conv2_2', conv2_2)
        pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')
       # tf.summary.image("pool2_2", pool2_2)
        pool2_2_transpose=tf.reshape(pool2_2, [1, 100, 37, 3])
        #tf.summary.image("pool2_2", pool2_2_transpose)
        pool2_2_transpose=tf.transpose(pool2_2, [3, 1, 2, 0])
        tf.summary.image("pool2_2", pool2_2_transpose,1)
        tf.summary.histogram('pool2_2', pool2_2)

        # Cross-Input Neighborhood Differences
        trans = tf.transpose(pool1_2, [0, 3, 1, 2])
        shape = trans.get_shape().as_list()
        m1s = tf.ones([shape[0], shape[1], shape[2], shape[3], 5, 5])
        reshape = tf.reshape(trans, [shape[0], shape[1], shape[2], shape[3], 1, 1])
        f = tf.multiply(reshape, m1s)

        trans = tf.transpose(pool2_2, [0, 3, 1, 2])
        reshape = tf.reshape(trans, [1, shape[0], shape[1], shape[2], shape[3]])
        g = []
        pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
        for i in xrange(shape[2]):
            for j in xrange(shape[3]):
                g.append(pad[:,:,:,i:i+5,j:j+5])

        concat = tf.concat(g, axis=0)
        reshape = tf.reshape(concat, [shape[2], shape[3], shape[0], shape[1], 5, 5])
        g = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
        reshape1 = tf.reshape(tf.subtract(f, g), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
        reshape2 = tf.reshape(tf.subtract(g, f), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
        k1 = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name='k1')
        k2 = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name='k2')

        #print("reshape1: "+str(reshape1))
        #print("reshape2: "+str(reshape2))
        reshape1_transpose=tf.transpose(reshape1, [1, 2, 3, 0])
        tf.summary.image("reshape1", reshape1_transpose,1)
        reshape2_transpose=tf.transpose(reshape2, [1, 2, 3, 0])
        tf.summary.image("reshape2", reshape2_transpose,1)

        # Patch Summary Features
        l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
        l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

        print("l1 : "+str(l1)+" l2 : "+str(l2))
        l1_transpose=tf.transpose(l1, [3, 1, 2, 0])
        tf.summary.image("l1", l1_transpose,1)
        l2_transpose=tf.transpose(l2, [3, 1, 2, 0])
        tf.summary.image("l2", l2_transpose,1)

        # Across-Patch Features
        m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
        pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
        m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
        pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

        print("poolm1: "+str(m1)+" poolm2 : "+str(m2))
        pool_m1_transpose=tf.transpose(pool_m1, [3, 1, 2, 0])
        tf.summary.image("pool_m1", pool_m1_transpose,1)
        pool_m2_transpose=tf.transpose(pool_m2, [3, 1, 2, 0])
        tf.summary.image("pool_m2", pool_m2_transpose,1)

        # Higher-Order Relationships
        concat = tf.concat([pool_m1, pool_m2], axis=3)
        reshape = tf.reshape(concat, [FLAGS.batch_size, -1])
        fc1 = tf.layers.dense(reshape, 500, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 2, name='fc2')

        print("fc1 : "+str(fc1)+" fc2: "+str(fc2))
        #fc1_transpose=tf.transpose(fc1, [ 1, 0])
        #tf.summary.image("fc1", fc1_transpose,1)
        #fc2_transpose=tf.transpose(fc2, [ 1, 0])
        #tf.summary.image("fc2", fc2_transpose,1)
        tf.summary.histogram('fc1', fc1)
        tf.summary.histogram('fc2', fc2)

        return fc2

def cmpims(im1,im2,inference,images,is_train,sess):
        image1 = cv2.imread(im1)
        image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        image2 = cv2.imread(im2)
        image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        test_images = np.array([image1, image2])

        feed_dict = {images: test_images, is_train: False}
        prediction = sess.run(inference, feed_dict=feed_dict)
#        return (bool(not np.argmax(prediction[0])))
        return prediction[0][0]

def findNextMax(scores,threshold):
    tempscore=scores
    index=-1
    m=0.0
    for s in range(len(scores)):
        if scores[s]<threshold and scores[s]>m:
            index=s
            m=scores[s]
    return index

def main(argv=None):
    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='labels')
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    tarin_num_id = 0
    val_num_id = 0

    if FLAGS.mode == 'train':
        tarin_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'train')
    elif FLAGS.mode == 'val':
        val_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'val')
    images1, images2 = preprocess(images, is_train)

    print('Build network')
    logits = network(images1, images2, weight_decay)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    inference = tf.nn.softmax(logits)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    train = optimizer.minimize(loss, global_step=global_step)
    lr = FLAGS.learning_rate

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            saver.restore(sess, ckpt.model_checkpoint_path)

        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in xrange(step, FLAGS.max_steps + 1):
                batch_images, batch_labels = cuhk03_dataset.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {learning_rate: lr, images: batch_images,
                    labels: batch_labels, is_train: True}
                sess.run(train, feed_dict=feed_dict)
                train_loss = sess.run(loss, feed_dict=feed_dict)
                print('Step: %d, Learning rate: %f, Train loss: %f' % (i, lr, train_loss))

                lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 1000 == 0:
                    saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)
        elif FLAGS.mode == 'val':
            total = 0.
            for _ in xrange(10):
                batch_images, batch_labels = cuhk03_dataset.read_data(FLAGS.data_dir, 'val', val_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {images: batch_images, labels: batch_labels, is_train: False}
                prediction = sess.run(inference, feed_dict=feed_dict)
                prediction = np.argmax(prediction, axis=1)
                label = np.argmax(batch_labels, axis=1)

                for i in xrange(len(prediction)):
                    if prediction[i] == label[i]:
                        total += 1
            print('Accuracy: %f' % (total / (FLAGS.batch_size * 10)))

            '''
            for i in xrange(len(prediction)):
                print('Prediction: %s, Label: %s' % (prediction[i] == 0, labels[i] == 0))
                image1 = cv2.cvtColor(batch_images[0][i], cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(batch_images[1][i], cv2.COLOR_RGB2BGR)
                image = np.concatenate((image1, image2), axis=1)
                cv2.imshow('image', image)
                key = cv2.waitKey(0)
                if key == 1048603:  # ESC key
                    break
            '''
        elif FLAGS.mode == 'test':
#            image_path1='20180516_155201_CH05_pic_recog/person/'
 #           image_path2='../persons/'#'20180516_155203_CH19_pic_recog/person/'
#            save_path1='20180516_sameperson/CH05/'
 #           save_path2='20180718/Preview_192.168.7.27_0_20180718_220140_3056687/'#'20180516_sameperson/CH19/'
#            if not os.path.exists(save_path1):
#                os.makedirs(save_path1)
  #          if not os.path.exists(save_path2):
   #             os.makedirs(save_path2)
#            image_files1=os.listdir(image_path1)
    #        image_files2=os.listdir(image_path2)
#            flen1=len(image_files1)
     #       flen2=len(image_files2)
#            print(flen1)
      #      print(flen2)

            print '--------end---------'


            image1 = cv2.imread(FLAGS.image1)
            image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            image2 = cv2.imread(FLAGS.image2)
            image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            test_images = np.array([image1, image2])

            merged_summary_op = tf.summary.merge_all()
            writer=tf.summary.FileWriter("./logs/",sess.graph)
          #  print("test_images : "+str(test_images))
            #image_shaped_input = tf.reshape(test_images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            #tf.summary.image('input1', image_shaped_input, 2)
    #        merged = tf.summary.merge_all()
 #           feed_dict = {images: test_images, is_train: False}
#            summary=sess.run(merged, feed_dict=feed_dict)
            feed_dict = {images: test_images, is_train: False}
            summary,prediction = sess.run([merged_summary_op,inference], feed_dict=feed_dict)
            print(bool(not np.argmax(prediction[0])))
            writer.add_summary(summary)
            writer.close()

if __name__ == '__main__':
    tf.app.run()
