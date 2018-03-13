import tensorflow as tf
import numpy as np



gph = tf.Graph()
with gph.as_default():
    with tf.variable_scope("inputs"):
        x = tf.placeholder(tf.float32,shape = [None,784],name = "input_image")
        y_actual = tf.placeholder(tf.float32,shape = [None,10],name = "true_cls")

    weights = [0]*5
    biases = [0]*5
    ls_dims = [784,256,128,64,32,10]

    a = x
    for i in range(5):
        with tf.variable_scope("layer"+str(i+1)):
            weights[i] = tf.Variable(tf.random_normal(shape = [ls_dims[i],ls_dims[i+1]],mean = 0.0, stddev=0.01))
            biases[i] = tf.Variable(tf.constant(1.0,shape = [ls_dims[i+1]]))
            a_norm = tf.contrib.layers.batch_norm(inputs = a,center = True,scale = True, is_training = True)
            z = tf.matmul(a_norm,weights[i]) + biases[i]
            a = tf.nn.relu(z)

    y_pred = tf.nn.softmax(logits = z,name = "pred_cls")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z,labels = y_actual),name = "loss")

    opt = tf.train.AdamOptimizer(learning_rate=0.1)
    train = opt.minimize(loss,name = "optimizer")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(60000):
        print("\rEpoch: ".format(i) + str(i),end = "")
