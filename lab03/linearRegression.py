import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight') 
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#가설 XW + b
hypothesis = X*W

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimize
learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y)*X)
descent = W -learning_rate*gradient
update = W.assign(descent)

sess = tf.Session()

#초기화 반드시 해야함
sess.run(tf.global_variables_initializer())

for step in range(21):
	sess.run(update, feed_dict={X: x_data, Y: y_data})
	print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
