# MNIST using convNN with different parameter initialization.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import argparse
import math

FLAGS = None

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Get initial stddev accordingly.
def GetStd(fan_in, fan_out):
	if FLAGS.fan_in_product > 0:
		return math.sqrt(FLAGS.fan_in_product / fan_in)
	elif FLAGS.fan_out_product > 0:
		return math.sqrt(FLAGS.fan_out_product / fan_out)
	else:
		return FLAGS.fix_std

# Construct Convolution layer
def ConvLayer(input_layer, weight_shape):
	assert len(weight_shape) == 4
	weight_std = GetStd(weight_shape[0] * weight_shape[1] * weight_shape[2], weight_shape[0] * weight_shape[1] * weight_shape[3])
	
	W = tf.Variable(tf.truncated_normal(weight_shape, stddev=weight_std), name='weights')
	b = tf.Variable(tf.constant(FLAGS.fix_biase, shape=[weight_shape[3]]), name='biases')
	
	h_conv = tf.nn.relu(tf.nn.conv2d(input_layer, W, strides=[1, 1, 1, 1], padding='SAME') + b)
	
	return h_conv
	
def FCLayer(input_layer, weight_shape):
	assert len(weight_shape) == 2
	weight_std = GetStd(weight_shape[0], weight_shape[1])
		
	W = tf.Variable(tf.truncated_normal(weight_shape, stddev=weight_std), name='weights')
	b = tf.Variable(tf.constant(FLAGS.fix_biase, shape=[weight_shape[1]]), name='biases')
		
	h_fc = tf.matmul(input_layer, W) + b

	return h_fc

def main():
	if tf.gfile.Exists(FLAGS.summaries_dir):
		tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
	tf.gfile.MakeDirs(FLAGS.summaries_dir)
	
	# Download and load MNIST data.
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True);

	sess = tf.InteractiveSession()
	
	# Build Graph
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, shape = [None, 784])
		y_ = tf.placeholder(tf.float32, shape = [None, 10])
		x_image = tf.reshape(x, [-1, 28, 28, 1])
	
	# Conv Layer 1: 28x28x1 -> 28x28x32
	with tf.name_scope('ConvLayer1'): 
		h_conv1 = ConvLayer(x_image, [5, 5, 1, 32])
	
	# Pool Layer 1: 28x28x32 -> 14 x 14 x 32
	with tf.name_scope('PoolLayer1'):
		h_pool1 = max_pool_2x2(h_conv1)
	
	# Conv Layer 2: 14x14x32 -> 14x14x64
	with tf.name_scope('ConvLayer2'): 
		h_conv2 = ConvLayer(h_pool1, [5, 5, 32, 64])
	
	# Pool Layer 2: 14x14x64 -> 7x7x64
	with tf.name_scope('PoolLayer2'):
		h_pool2 = max_pool_2x2(h_conv2)
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	
	# FC Layer 1: 7*7*64 -> 1024
	with tf.name_scope('FCLayer1'):
		h_fc1 = FCLayer(h_pool2_flat, [7*7*64, 1024])
		h_fc1 = tf.nn.relu(h_fc1)
	
	# Dropout
	with tf.name_scope('Dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	# FC Layer 2: 1024 -> 10
	with tf.name_scope('FCLayer2'):
		y_conv = FCLayer(h_fc1_drop, [1024, 10])
	
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	tf.scalar_summary('cross_entropy', cross_entropy)
	
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	tf.scalar_summary('accuracy', accuracy)

	#merged_summary = tf.summary.merge_all()
	merged_summary = tf.merge_all_summaries()

	train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
	test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
	
	#global_variables_initializers
	init = tf.initialize_all_variables()
	sess.run(init)

	for i in range(1000):
		if i % 50 == 0:
			feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
			summary, acc = sess.run([merged_summary, accuracy], feed_dict = feed_dict)
			test_writer.add_summary(summary, i)
			print "Test accuracy at step %s: %s" %(i, acc)
		else:
			batch = mnist.train.next_batch(50)
			feed_dict={x:batch[0], y_: batch[1], keep_prob: FLAGS.dropout}
			summary, _ = sess.run([merged_summary, train_step], feed_dict = feed_dict)
			train_writer.add_summary(summary, i)
	train_writer.close()
	test_writer.close()
	

if __name__ == '__main__':
	# Parse args
	parser= argparse.ArgumentParser()
	parser.add_argument('--fan_in_product', type=float, default = 0.0, help = 'Initialize weight according to fan-in')
	parser.add_argument('--fan_out_product', type=float, default = 0.0, help = 'Initialize weight according to fan-out')
	parser.add_argument('--fix_std', type=float, default = 0.1, help = 'Initialize weight using fixed std')
	
	parser.add_argument('--fix_biase', type=float, default = 0.1, help = 'Initialize using fixed biases')
	
	parser.add_argument('--dropout', type=float, default = 0.5, help = 'Dropout Ratio')
	
	parser.add_argument('--summaries_dir', type=str, default = './tmp', help = 'Directory to put log data')
	FLAGS, unparsed = parser.parse_known_args()
	
	main()

