import tensorflow as tf

#placeholder 특별한 노드
#placeholder 이용하면 나중에 feed_dict에서 값을 넘겨 줄 수 있다. 하나뿐만 아니라 여러가지를 넘길 수 도 있다.
#sess.run(opertion, feed_dict={x:x_data})
sess = tf.Session()
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b:[2,4]}))