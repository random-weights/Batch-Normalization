import tensorflow as tf
import numpy as np
import pandas as pd

class Data():

    def __init__(self):
        self.size = 0

    def setDataPaths(self,x_path,y_path):
        self.x_path = x_path
        self.y_path = y_path

    def getAllData(self):
        df_x = pd.read_csv(self.x_path,header = None,sep = ',')
        df_y = pd.read_csv(self.y_path, header=None, sep=',')
        self.size = len(df_y)
        self.x_data = np.array(df_x).astype(int)
        self.y_data = np.array(df_y).astype(int)

    def getRandomdata(self,batch_size=None):
        if batch_size is None:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

        rand_indices = np.random.choice(self.size,self.batch_size,replace = False)
        self.x_batch = self.x_data[rand_indices]
        self.y_batch = self.y_data[rand_indices]

batch_size = 128

x = tf.placeholder('float',shape = [batch_size,784])
y_actual = tf.placeholder('float',shape = [batch_size,10])
alpha = tf.placeholder('float')

weights = [0]*5
biases = [0]*5

ls_dims = [784,256,128,64,32,10]

#initialize all weights and biases
a = x
for i in range(5):
    weights[i] = tf.Variable(tf.random_normal(shape = [ls_dims[i],ls_dims[i+1]],mean = 0.0, stddev=0.01))
    biases[i] = tf.Variable(tf.constant(1.0,shape = [ls_dims[i+1]]))
    z = tf.matmul(a,weights[i]) + biases[i]
    a = tf.nn.relu(z)

y_pred = tf.nn.softmax(logits = z)
_,acc = tf.metrics.accuracy(tf.argmax(y_pred,axis = 1),tf.argmax(y_actual,axis = 1))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z,labels = y_actual))

opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train = opt.minimize(loss)

epochs = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_data = Data()
    train_data.setDataPaths(x_path = "data/x_train.csv",y_path="data/y_train.csv")
    train_data.getAllData()
    ls_fname = ["basic_run_"+str(i)+".csv" for i in range(1,6)]
    ls_fh = [open(file,'a') for file in ls_fname]
    ls_alpha = [0.001,0.005,0.01,0.05,0.1]
    for a in range(5):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("----------------------------Run: ",a)
        for i in range(epochs):
            train_data.getRandomdata(batch_size)
            x_batch = train_data.x_batch
            y_batch = train_data.y_batch
            feed_dict = {x: x_batch,y_actual:y_batch,alpha:ls_alpha[a]}
            _,accuracy = sess.run([train,acc],feed_dict)
            final_str = str(i) + "," + str(accuracy * 100)
            print("Epoch: ",i,"\tAccuracy: ",accuracy*100)
            print(final_str,file = ls_fh[a])

    for handler in ls_fh:
        handler.close()
