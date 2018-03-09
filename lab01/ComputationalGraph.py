import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1:", node1, "node2:", node2)
print("node3:", node3) #이렇게 값을 넣으면 올바른 값이 출력되지 않는다. Session()을 만들고 run해야 한다

sess = tf.Session() #Session함수를 이용해서 객체를 생성하고 그 객체의 run함수의 인자로 node1 과 node2를 넣는다.
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run: ", sess.run(node3))