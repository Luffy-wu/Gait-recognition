

```python
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
```

    /Users/luffy/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


    WARNING:tensorflow:From /Users/luffy/anaconda3/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py:118: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.
    step 0, test accuracy: 0.0254042, loss: 20.8801
    step 100, test accuracy: 0.344111, loss: 2.77647
    step 200, test accuracy: 0.52194, loss: 1.80824
    step 300, test accuracy: 0.622402, loss: 1.41782
    step 400, test accuracy: 0.677829, loss: 1.19016
    step 500, test accuracy: 0.720554, loss: 1.0777
    step 600, test accuracy: 0.756351, loss: 0.938708
    step 700, test accuracy: 0.756351, loss: 0.887941
    step 800, test accuracy: 0.774827, loss: 0.841865
    step 900, test accuracy: 0.809469, loss: 0.754522
    step 1000, test accuracy: 0.823326, loss: 0.708289
    step 1100, test accuracy: 0.831409, loss: 0.677985
    step 1200, test accuracy: 0.838337, loss: 0.638494
    step 1300, test accuracy: 0.842956, loss: 0.624689
    step 1400, test accuracy: 0.844111, loss: 0.626332
    step 1500, test accuracy: 0.844111, loss: 0.612932
    step 1600, test accuracy: 0.854503, loss: 0.569524
    step 1700, test accuracy: 0.854503, loss: 0.565087
    step 1800, test accuracy: 0.867206, loss: 0.550617
    step 1900, test accuracy: 0.862587, loss: 0.549755
    step 2000, test accuracy: 0.876443, loss: 0.510148
    step 2100, test accuracy: 0.872979, loss: 0.499258
    step 2200, test accuracy: 0.881062, loss: 0.485307
    step 2300, test accuracy: 0.874134, loss: 0.514515
    step 2400, test accuracy: 0.878753, loss: 0.507032
    step 2500, test accuracy: 0.875289, loss: 0.492049
    step 2600, test accuracy: 0.886836, loss: 0.444225
    step 2700, test accuracy: 0.881062, loss: 0.447543
    step 2800, test accuracy: 0.886836, loss: 0.445326
    step 2900, test accuracy: 0.89261, loss: 0.435554
    step 3000, test accuracy: 0.891455, loss: 0.433298
    step 3100, test accuracy: 0.887991, loss: 0.450135
    step 3200, test accuracy: 0.891455, loss: 0.438998
    step 3300, test accuracy: 0.904157, loss: 0.405205
    step 3400, test accuracy: 0.8903, loss: 0.479928
    step 3500, test accuracy: 0.907621, loss: 0.401645
    step 3600, test accuracy: 0.901848, loss: 0.434153
    step 3700, test accuracy: 0.903002, loss: 0.421022
    step 3800, test accuracy: 0.904157, loss: 0.439091
    step 3900, test accuracy: 0.905312, loss: 0.428991
    step 4000, test accuracy: 0.911085, loss: 0.40763
    step 4100, test accuracy: 0.909931, loss: 0.419832
    step 4200, test accuracy: 0.91455, loss: 0.403698
    step 4300, test accuracy: 0.916859, loss: 0.377179
    step 4400, test accuracy: 0.91224, loss: 0.414692
    step 4500, test accuracy: 0.907621, loss: 0.425325
    step 4600, test accuracy: 0.921478, loss: 0.394585
    step 4700, test accuracy: 0.916859, loss: 0.396334
    step 4800, test accuracy: 0.915704, loss: 0.41014
    step 4900, test accuracy: 0.907621, loss: 0.426154
    step 5000, test accuracy: 0.921478, loss: 0.394869
    step 5100, test accuracy: 0.908776, loss: 0.42857
    step 5200, test accuracy: 0.918014, loss: 0.417845
    step 5300, test accuracy: 0.919169, loss: 0.400205
    step 5400, test accuracy: 0.918014, loss: 0.407162
    step 5500, test accuracy: 0.920323, loss: 0.402868
    step 5600, test accuracy: 0.909931, loss: 0.451346
    step 5700, test accuracy: 0.909931, loss: 0.459315
    step 5800, test accuracy: 0.915704, loss: 0.443558
    step 5900, test accuracy: 0.919169, loss: 0.416192
    step 6000, test accuracy: 0.918014, loss: 0.437447
    step 6100, test accuracy: 0.924942, loss: 0.410006
    step 6200, test accuracy: 0.919169, loss: 0.442774
    step 6300, test accuracy: 0.91224, loss: 0.471452
    step 6400, test accuracy: 0.919169, loss: 0.407574
    step 6500, test accuracy: 0.920323, loss: 0.417277
    step 6600, test accuracy: 0.918014, loss: 0.422551
    step 6700, test accuracy: 0.924942, loss: 0.400044
    step 6800, test accuracy: 0.919169, loss: 0.454586
    step 6900, test accuracy: 0.922633, loss: 0.404275
    step 7000, test accuracy: 0.924942, loss: 0.462554
    step 7100, test accuracy: 0.923788, loss: 0.429201
    step 7200, test accuracy: 0.919169, loss: 0.461661
    step 7300, test accuracy: 0.931871, loss: 0.395814
    step 7400, test accuracy: 0.923788, loss: 0.442353
    step 7500, test accuracy: 0.929561, loss: 0.395391
    step 7600, test accuracy: 0.913395, loss: 0.522306
    step 7700, test accuracy: 0.928406, loss: 0.415137
    step 7800, test accuracy: 0.921478, loss: 0.45981
    step 7900, test accuracy: 0.929561, loss: 0.439067
    step 8000, test accuracy: 0.931871, loss: 0.429213
    step 8100, test accuracy: 0.922633, loss: 0.447705
    step 8200, test accuracy: 0.913395, loss: 0.485711
    step 8300, test accuracy: 0.933025, loss: 0.42576
    step 8400, test accuracy: 0.920323, loss: 0.496535
    step 8500, test accuracy: 0.927252, loss: 0.458236
    step 8600, test accuracy: 0.926097, loss: 0.445187
    step 8700, test accuracy: 0.929561, loss: 0.416223
    step 8800, test accuracy: 0.93418, loss: 0.424454
    step 8900, test accuracy: 0.919169, loss: 0.504584
    step 9000, test accuracy: 0.922633, loss: 0.476162
    step 9100, test accuracy: 0.926097, loss: 0.449849
    step 9200, test accuracy: 0.929561, loss: 0.393292
    step 9300, test accuracy: 0.930716, loss: 0.43126
    step 9400, test accuracy: 0.93649, loss: 0.433651
    step 9500, test accuracy: 0.933025, loss: 0.413674
    step 9600, test accuracy: 0.929561, loss: 0.471349
    step 9700, test accuracy: 0.929561, loss: 0.487002
    step 9800, test accuracy: 0.926097, loss: 0.521799
    step 9900, test accuracy: 0.915704, loss: 0.567571
    best test accuracy: 0.93649
    final test accuracy 0.933025



```python
print("final test accuracy %g"%test_accuracy)
```

    final test accuracy 0.915704



```python
!tensorboard --logdir='/Users/luffy/Desktop/gait_test/tensorboard/'
```

    /Users/luffy/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    TensorBoard 1.6.0 at http://lixiangwus-MacBook-Air.local:6006 (Press CTRL+C to quit)
    ^C



```python
http://localhost:6006
```
