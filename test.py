import tensorflow as tf
import numpy as np

x_data = np.array(
    [[1, 1, 4, 50], [1, 0, 4, 5], [1, 0, 4, 20], [1, 0, 4, 10], [1, 1, 4, 100], [1, 0, 4, 12],
     [1, 0, 4, 15], [1, 1, 4, 75]]
)

y_data = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W1 = tf.Variable(tf.random_uniform([4, 4], -1., 1.))
b1 = tf.Variable(tf.zeros([4]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(tf.random_normal([4, 5]))
b2 = tf.Variable(tf.zeros([5]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.Variable(tf.random_normal([5, 6]))
b3 = tf.Variable(tf.zeros([6]))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

W4 = tf.Variable(tf.random_normal([6, 4]))
b4 = tf.Variable(tf.zeros([4]))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))

W5 = tf.Variable(tf.random_normal([4, 3]))
b5 = tf.Variable(tf.zeros([3]))
L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), b5))

W6 = tf.Variable(tf.random_normal([3, 3]))
b6 = tf.Variable(tf.zeros([3]))
model = tf.add(tf.matmul(L5, W6), b6)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
sess.close()
