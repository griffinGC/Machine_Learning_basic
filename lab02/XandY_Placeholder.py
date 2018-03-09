import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]), name='weight') #Variable은 기존의 변수와는 약간 다른개념이다. 텐서플로우가 사용하는 variable
b = tf.Variable(tf.random_normal([1]), name = 'bias') #텐서플로우 실행시키면 자체적으로 실행하는 함수. 혹은 trainable 학습하는 과정에서 변경시킨다.

#가설 XW + b
hypothesis = x*w+ b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))
#tf.square는 제곱 시키는 함수
#tf.reduce_mean 은 평균내어주는 함수

#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict={x : [1, 2, 3], y : [1, 2, 3]})
		
	if step%20 == 0:
		print(step, cost_val, w_val, b_val)
