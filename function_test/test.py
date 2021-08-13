# Create a mixture of two Gaussians:
import tensorflow_probability as tfp
import tensorflow as tf

x = tf.reshape(tf.range(12), (3, 2, 2))

p, q = tf.unstack(x, axis=2)
p.shape.as_list()

print(x)

print(p)
print(q)
