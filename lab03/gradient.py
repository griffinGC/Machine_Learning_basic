import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.0)


#가설 XW + b
hypothesis = X*W

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

sess = tf.Session()

#초기화 반드시 해야함
sess.run(tf.global_variables_initializer())

for step in range(21):
	print(step, sess.run(W))
	sess.run(train)
