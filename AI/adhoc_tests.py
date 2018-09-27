import tensorflow as tf
import tensorboard

tf.reset_default_graph()
# create graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
# launch the graph in a session
with tf.Session() as sess:
    print(sess.run(c))
    writer = tf.summary.FileWriter('./graphs', sess.graph)

tensorboard --logdir='C:\Users\sbayed\Desktop\sbayed_local\Machine Learning\AI\AI\graphs'