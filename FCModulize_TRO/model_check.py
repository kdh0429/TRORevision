import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()

new_saver = tf.compat.v1.train.import_meta_graph('model/model.ckpt.meta')
new_saver.restore(sess, 'model/model.ckpt')

graph = tf.compat.v1.get_default_graph()
name = [n.name for n in tf.compat.v1.trainable_variables()]
for n in name:
    print_value = graph.get_tensor_by_name(n)
    print(n, sess.run(print_value))