import tensorflow as tf
import numpy as np
import cv2
import cuhk03_dataset

import os
import sys
import shutil
import math

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
        # Tied Convolution
        conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_1')
        pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
        conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_2')
        pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')
        conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_1')
        pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
        conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_2')
        pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')

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

        # Patch Summary Features
        l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
        l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

        # Across-Patch Features
        m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
        pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
        m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
        pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

        # Higher-Order Relationships
        concat = tf.concat([pool_m1, pool_m2], axis=3)
        reshape = tf.reshape(concat, [FLAGS.batch_size, -1])
        fc1 = tf.layers.dense(reshape, 500, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 2, name='fc2')

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
        print ('-----')
        print (prediction[0])
        print ('=====')
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
            image_path2='../persons/'#'tmppersons_crop/'#'../persons/37-2/'#'20180516_155203_CH19_pic_recog/person/'
#            save_path1='20180516_sameperson/CH05/'
            save_path2='20181109/44/'#'tmps/tmpsame/'#'20180911/37-2/'#'20180718/Preview_192.168.7.27_0_20180718_220140_3056687/'#'20180516_sameperson/CH19/'
#            if not os.path.exists(save_path1):
#                os.makedirs(save_path1)
            if not os.path.exists(save_path2):
                os.makedirs(save_path2)
#            image_files1=os.listdir(image_path1)
            image_files2=os.listdir(image_path2)
#            flen1=len(image_files1)
            flen2=len(image_files2)
#            print(flen1)
            print(flen2)
#            flags1=[0]*(flen1)
#            flags2=[0]*(flen2)

#            index1_1=1#4115#1
#            maxindex1=9230#4116#9230
#            initpath1='20180516_155201_CH05_pic_recog/initperson/'
#            init_persons1=os.listdir(initpath1)
#            initp1={}
#            for i in range(len(init_persons1)):
#                initp1[i]=initpath1+init_persons1[i]
#                new_path=save_path1+str(i)
#                if not os.path.exists(new_path):
#                    os.makedirs(new_path)
#                shutil.copyfile(initpath1+init_persons1[i],new_path+'/'+init_persons1[i])
#            if not os.path.exists(save_path1+'nones'):
#                os.makedirs(save_path1+'nones')
#            while index1_1<maxindex1:
#                maxindexes=[]
#                maxscores=[]
#                im2_names=[]
#                for j in range(flen1):
#                    index1_2=int(image_files1[j].split('.jpg')[0].split('_')[-1])
#                    if index1_2==(index1_1+1):
#                       im2_names.append(image_files1[j])
#                for nm2 in range(len(im2_names)):
#                   im2=image_path1+im2_names[nm2]
#                   im2_scores={}
#                   maxindex=-1
#                   maxscore=0.0
#                   for nm1 in range(len(initp1)):
#                       im1=initp1[nm1]
#                       y1_1=int(im1.split('_')[-6])
#                       x1_1=int(im1.split('_')[-5])
#                       y1_2=int(im1.split('_')[-4])
#                       x1_2=int(im1.split('_')[-3])
#                       y1_0=(y1_1+y1_2)/2.0
#                       x1_0=(x1_1+x1_2)/2.0
#                       y2_1=int(im2.split('_')[-6])
#                       x2_1=int(im2.split('_')[-5])
#                       y2_2=int(im2.split('_')[-4])
#                       x2_2=int(im2.split('_')[-3])
#                       y2_0=(y2_1+y2_2)/2.0
#                       x2_0=(x2_1+x2_2)/2.0
#                       dist=math.sqrt(math.pow(y1_0-y2_0,2)+math.pow(x1_0-x2_0,2))
#                       predtemp=0.0
#                       if dist>50:
#                           predtemp=0.0
#                       else:
#                           predtemp=cmpims(im1,im2,inference,images,is_train,sess)
#                       im2_scores[nm1]=predtemp
#                       if predtemp>maxscore and predtemp>0.9:
#                          maxindex=nm1
#                          maxscore=predtemp
#                   maxindexes.append(maxindex)
#                   maxscores.append(im2_scores)
#                while True:
#                    moreindexes=[]
#                    for a in maxindexes:
#                        if maxindexes.count(a)>1  and (not a in moreindexes):
#                            moreindexes.append(a)
#                    if len(moreindexes)==0:
#                        break
#                    if len(moreindexes)==1 and moreindexes[0]==-1:
#                        break

#                    for a in moreindexes:
#                        flag=0
#                        list_index=[]
#                        for n in range(maxindexes.count(a)):
#                            sec=flag
#                            flag=maxindexes[flag:].index(a)
#                            list_index.append(flag+sec)
#                            flag=list_index[-1:][0]+1
                        
#                        print str(a)+" : "+str(list_index)
#                        temp_index=list_index[0]
#                        for m in list_index:
#                            if(maxscores[m].get(a)>maxscores[temp_index].get(a)):
#                                temp_index=m
#                        for m in list_index:
#                            if m!=temp_index:  
#                                nextindex=findNextMax(maxscores[m],maxscores[m].get(a))
#                                if nextindex==-1:
#                                    maxindexes[m]=-1
#                                elif maxscores[m][nextindex] < 0.9:
#                                    maxindexes[m]=-1
#                                else:
#                                    maxindexes[m]=nextindex
#                for m in range(len(maxindexes)):
#                    maxindex=maxindexes[m]
    
#                    if maxindex!=-1:
#                       shutil.copyfile(image_path1+im2_names[m],save_path1+str(maxindex)+'/'+im2_names[m])
#                       initp1[maxindex]=save_path1+str(maxindex)+'/'+im2_names[m]
#                    else:
#                       shutil.copyfile(image_path1+im2_names[m],save_path1+'nones/'+im2_names[m]) 
#                index1_1=index1_1+1

            index2_1=1#1
            maxindex2=358#274#358#5260#10422#10422
            initpath2='1109initperson/4/'#'tmpinitperson/'#'initperson/37-2/'#'tmpinitperson/'#'initperson/'#'20180516_155203_CH19_pic_recog/initperson/'
            init_persons2=os.listdir(initpath2)
            initp2={}
            for i in range(len(init_persons2)):
                initp2[i]=initpath2+init_persons2[i]
                new_path=save_path2+str(i)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(initpath2+init_persons2[i],new_path+'/'+init_persons2[i])
            if not os.path.exists(save_path2+'nones'):
                os.makedirs(save_path2+'nones')
            while index2_1<maxindex2:
                print(index2_1)
                maxindexes=[]
                maxscores=[]
                im2_names=[]
                for j in range(flen2):
                    index2_2=int(image_files2[j].split('.jpg')[0].split('-')[-1])
                    if index2_2==(index2_1+1):
                       im2_names.append(image_files2[j])
                for nm2 in range(len(im2_names)):
                   im2=image_path2+im2_names[nm2]
                   im2_scores={}
                   maxindex=-1
                   maxscore=0.0
                   for nm1 in range(len(initp2)):
                       im1=initp2[nm1]
                       y1_1=int(im1.split('_')[-6])
                       x1_1=int(im1.split('_')[-5])
                       y1_2=int(im1.split('_')[-4])
                       x1_2=int(im1.split('_')[-3])
                       y1_0=(y1_1+y1_2)/2.0
                       x1_0=(x1_1+x1_2)/2.0
                       y2_1=int(im2.split('_')[-6])
                       x2_1=int(im2.split('_')[-5])
                       y2_2=int(im2.split('_')[-4])
                       x2_2=int(im2.split('_')[-3])
                       y2_0=(y2_1+y2_2)/2.0
                       x2_0=(x2_1+x2_2)/2.0
                       dist=math.sqrt(math.pow(y1_0-y2_0,2)+math.pow(x1_0-x2_0,2))
                       predtemp=0.0
                       if dist>150:#50##################
                           predtemp=0.0
                       else:
                           predtemp=cmpims(im1,im2,inference,images,is_train,sess)
                       im2_scores[nm1]=predtemp
                      # print "im1: "+str(im1)+" im2: "+str(im2)
                      # print "predtemp: "+str(predtemp)
                       if predtemp>maxscore and predtemp>0.4:#0.9#################################
                          maxindex=nm1
                          maxscore=predtemp
                   maxindexes.append(maxindex)
                   maxscores.append(im2_scores)
                while True:
                    moreindexes=[]
                    for a in maxindexes:
                        if maxindexes.count(a)>1  and (not a in moreindexes):
                            moreindexes.append(a)
                    if len(moreindexes)==0:
                        break
                    if len(moreindexes)==1 and moreindexes[0]==-1:
                        break
                    for a in moreindexes:
                        flag=0
                        list_index=[]
                        for n in range(maxindexes.count(a)):
                            sec=flag
                            flag=maxindexes[flag:].index(a)
                            list_index.append(flag+sec)
                            flag=list_index[-1:][0]+1
                        print str(a)+" : "+str(list_index)
                        temp_index=list_index[0]
                        for m in list_index:
                            if(maxscores[m].get(a)>maxscores[temp_index].get(a)):
                                temp_index=m
                        for m in list_index:
                            if m!=temp_index:
                                nextindex=findNextMax(maxscores[m],maxscores[m].get(a))
                                if nextindex==-1:
                                    maxindexes[m]=-1
                                elif maxscores[m][nextindex] < 0.9:
                                    maxindexes[m]=-1
                                else:
                                    maxindexes[m]=nextindex
                for m in range(len(maxindexes)):
                    maxindex=maxindexes[m]
                    if maxindex!=-1:
                       shutil.copyfile(image_path2+im2_names[m],save_path2+str(maxindex)+'/'+im2_names[m])
                       initp2[maxindex]=save_path2+str(maxindex)+'/'+im2_names[m]
                    else:
                       shutil.copyfile(image_path2+im2_names[m],save_path2+'nones/'+im2_names[m])
                index2_1=index2_1+1

            print '--------end---------'


#            image1 = cv2.imread(FLAGS.image1)
 #           image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
  #          image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
   #         image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
    #        image2 = cv2.imread(FLAGS.image2)
     #       image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
      #      image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
       #     image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        #    test_images = np.array([image1, image2])

         #   feed_dict = {images: test_images, is_train: False}
          #  prediction = sess.run(inference, feed_dict=feed_dict)
           # print(bool(not np.argmax(prediction[0])))

if __name__ == '__main__':
    tf.app.run()
