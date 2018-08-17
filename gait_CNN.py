"""
Created on Fri Apr 27 20:51:42 2018

@author: luffy
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
tf.reset_default_graph()

####################Load data############################

#Data preprocessing

def data_normalize(data_temp):
    data_temp2=data_temp.T
    data_temp2 -=np.mean(data_temp2,axis=0)
    data_temp2 /=np.std(data_temp2,axis=0)
    data_temp=data_temp2.T
    return data_temp

def Xload(paths):
    data_list=[]
    for path in paths:
        temp=data_normalize(np.loadtxt(path))
        data_list.append(temp)
    ######合并acc_x/y/z,gyr_x/y/z
    lenth=len(data_list[0])
    width=len(data_list[0][1])
    data_np=np.empty((lenth,width*6))
    for i in range(width*6):
        temp= i%6
        data_np[:,i]=data_list[temp][:,i//6]
    return data_np

def Yload(path):
    train_np= np.loadtxt(path)  
    train_np1=list(train_np)
    train_np2=list(map(int, train_np1))
    ytrain_np=[train_value-1 for train_value in train_np2]
    #Convert tags to onehot
    batch_size = tf.size(ytrain_np)
    labels = tf.expand_dims(ytrain_np, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels],1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 21]), 1.0, 0.0)
    return onehot_labels.eval()
    
input_types= ["acc_x.txt","acc_y.txt","acc_z.txt","gyr_x.txt","gyr_y.txt","gyr_z.txt"]
train_paths=["/Users/luffy/Desktop/gait_test/train/train_"+ i for i in input_types]
train_np=Xload(train_paths)
test_paths=["/Users/luffy/Desktop/gait_test/test/test_"+ i for i in input_types]
test_np=Xload(test_paths)


sess = tf.InteractiveSession()
ytrain_path= "/Users/luffy/Desktop/gait_test/train/y_train.txt"
ytrain_label=Yload(ytrain_path)
ytest_path= "/Users/luffy/Desktop/gait_test/test/y_test.txt"
ytest_label=Yload(ytest_path)





####################Set parameters###############################
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



x = tf.placeholder(tf.float32, [None, 128*6])  
x_image = tf.reshape(x, [-1,16,8,6]) #reshape

"""
# first layer
"""
W_conv1 = weight_variable([5, 5, 6, 32])  
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
# second layer
"""
W_conv2 = weight_variable([5, 5, 32, 64]) 
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


"""
# third layer, the fully connected layer with an input dimension of 4*2*64 and an output dimension of 1024
"""
W_fc1 = weight_variable([4*2*64, 1024])  
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
"""
# fourth layer，with an input dimension of 1024 and an output dimension of 21，corresponding to 21 categories
"""
W_fc2 = weight_variable([1024, 21])
b_fc2 = bias_variable([21])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #Use softmax as a multi-class activation function
y_ = tf.placeholder(tf.float32, [None, 21])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # Loss function，cross entropy
tf.summary.scalar('loss', cross_entropy)
train_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # Use Adam optimizer
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
log_dir='/Users/luffy/Desktop/gait_test/tensorboard/'  # Use tensorboard
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir+'test/')
sess.run(tf.initialize_all_variables())  # Variable initialization
optimal_accuracy = 0.0
#train
for i in range(10000):  
    batch_x,batch_y = next_batch(50,train_np,ytrain_label)  
    if i%100 == 0:
        loss,test_accuracy,test_summary=sess.run([cross_entropy,accuracy,merged],feed_dict={
            x:test_np, y_: ytest_label, keep_prob: 1.0})
        test_writer.add_summary(test_summary, i)
        print("step %d, test accuracy: %g, loss: %g"%(i, test_accuracy,loss))
        optimal_accuracy = max(optimal_accuracy, test_accuracy)
    summary_, _ = sess.run([merged,train_optimizer],feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})   
    train_writer.add_summary(summary_, i)
train_writer.close()
test_writer.close()

print("best test accuracy: %g"%(optimal_accuracy))
print("final test accuracy %g"%accuracy.eval(feed_dict={x: test_np, y_: ytest_label, keep_prob: 1.0}))
#!tensorboard --logdir='/Users/luffy/Desktop/gait_test/tensorboard/'
