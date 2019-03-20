from dataGenerator import DataGenerator
import matplotlib.pyplot as plt
from model import Model
import tensorflow as tf
from utils import loss
import numpy as np


def func(x):
    y = np.zeros(x.shape)
    y[x < 1] = x[x < 1]
    y[x >= 1] = 4 * x[x >= 1] - 3
    y[x <= 0] = -2 * x[x <= 0]
    return y


epoch = 10000
d = DataGenerator(100, (0, 0.05), np.sin)
x, y = d()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([-10.5, 10.5])
plt.ylim([-1.5, 1.5])
ax.scatter(x, y)
plt.ion()
print(d)
model = Model([1, 10, 1], [tf.nn.sigmoid, None])
x_ = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])
y_pred = model(x_)
mse = loss(y_pred, y_)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(mse)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        _, Mse, y_prediction = sess.run([train, mse, y_pred], feed_dict={x_: x, y_: y})
        if i % 50 == 0:
            plt.pause(0.1)
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            print('epochs {} loss {}'.format(i, Mse))
            lines = ax.plot(x, y_prediction)
plt.ioff()
plt.show()
