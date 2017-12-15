import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class net():

	def __init__(self):
		self.hiddenSz1 = 50
		self.hiddenSz2 = 25
		self.keepPrb1 = 0.9
		self.keepPrb2 = 0.3
		self.learning_rate = 0.001
		self.sizeOfTupleInput = 1
		self.batchSz = 1
		self.trainingEpochs = 1
		self.outputSz = 1

	def train(self, inny, ansy, label):
		inpt, ans, W1, b1, l1Out, L1Out, L1Drop, W2, b2, l2Out, L2Out, L2Drop, \
		 W3, b3, out, loss, sgd, train_op = self.model2(self.batchSz)

		saver = tf.train.Saver()

		session = tf.Session()
		session.run(tf.global_variables_initializer())
		print "Training the", label
		numbBatches = int(np.floor(len(inny)/self.batchSz))
		for j in range(self.trainingEpochs):
			print "Training Epoch: ", j
			avgLoss = 0.0
			for i in xrange(numbBatches):
				lowInd = i*self.batchSz
				diction = {inpt: inny[lowInd:lowInd+self.batchSz,:], ans: ansy[lowInd:lowInd+self.batchSz]}
				_, losses = session.run([train_op, loss], feed_dict=diction)
				avgLoss += losses[0]
				if (i%(8000/self.batchSz) == 0):
					print "Percent complete: ", int((i/float(numbBatches))*100)
			print "Loss: ", avgLoss/(numbBatches*self.batchSz)

		save_path = saver.save(session, "./weights/"+label)

	def eval(self, inny, ansy, label):
		tf.reset_default_graph()

		inpt, ans, W1, b1, l1Out, L1Out, L1Drop, W2, b2, l2Out, L2Out, L2Drop, \
		 W3, b3, out, loss, sgd, train_op = self.model2(len(inny))

		saver = tf.train.Saver()

		sess = tf.Session()

		saver.restore(sess, "./weights/"+label)
		print "Evaluating the", label
		diction = {inpt: inny, ans: ansy}
		#avgLoss = lossOut/len(inny)
		#print("Avg dist from truth : %s" % avgLoss)
		return sess.run(out, diction)

	def model1(self, size):
		inpt = tf.placeholder(tf.float32, [size, self.sizeOfTupleInput])
		ans = tf.placeholder(tf.float32, [size])

		W1 = tf.Variable(tf.random_normal([self.sizeOfTupleInput, self.hiddenSz1], stddev=.1))
		b1 = tf.Variable(tf.random_normal([self.hiddenSz1], stddev=.1))
		l1Out = tf.matmul(inpt, W1) + b1
		L1Out = tf.nn.relu(l1Out)
		L1OutDrop = tf.nn.dropout(L1Out, self.keepPrb1)
		W2 = tf.Variable(tf.random_normal([self.hiddenSz1, self.outputSz], stddev=.1))
		b2 = tf.Variable(tf.random_normal([self.outputSz], stddev=.1))
		out = tf.nn.sigmoid(tf.matmul(L1OutDrop,W2)+b2)

		#loss corresponds to euclid dist
		loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(ans, out)), 1)) 
		sgd = tf.train.AdamOptimizer(self.learning_rate)
		train_op = sgd.minimize(loss)
		return [inpt, ans, W1, b1, l1Out, L1Out, L1OutDrop, W2, b2, out, loss, sgd, train_op]


	def model2(self, size):
		inpt = tf.placeholder(tf.float32, [size, self.sizeOfTupleInput])
		ans = tf.placeholder(tf.float32, [size])

		W1 = tf.Variable(tf.random_normal([self.sizeOfTupleInput, self.hiddenSz1], stddev=.1))
		b1 = tf.Variable(tf.random_normal([self.hiddenSz1], stddev=.1))
		l1Out = tf.matmul(inpt, W1) + b1
		L1Out = tf.nn.relu(l1Out)
		L1Drop = tf.nn.dropout(L1Out, self.keepPrb1)
		W2 = tf.Variable(tf.random_normal([self.hiddenSz1, self.hiddenSz2], stddev=.1))
		b2 = tf.Variable(tf.random_normal([self.hiddenSz2], stddev=.1))
		l2Out = tf.matmul(L1Drop, W2) + b2
		L2Out = tf.nn.relu(l2Out)
		L2Drop = tf.nn.dropout(L2Out, self.keepPrb2)
		W3 = tf.Variable(tf.random_normal([self.hiddenSz2, self.outputSz], stddev=.1))
		b3 = tf.Variable(tf.random_normal([self.outputSz], stddev=.1))
		out = tf.nn.sigmoid(tf.matmul(L2Drop,W3)+b3)

		#loss corresponds to euclid dist
		loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(ans, out)), 1)) 
		sgd = tf.train.AdamOptimizer(self.learning_rate)
		train_op = sgd.minimize(loss)
		return [inpt, ans, W1, b1, l1Out, L1Out, L1Drop, W2, b2, l2Out, L2Out, L2Drop, \
		 W3, b3, out, loss, sgd, train_op]




class netIm():

	def __init__(self):
		self.hiddenSz1 = 50
		self.hiddenSz2 = 25
		self.keepPrb1 = 0.9
		self.keepPrb2 = 0.3
		self.learning_rate = 0.001
		self.sizeOfTupleInput = 1
		self.batchSz = 1
		self.trainingEpochs = 1
		self.outputSz = 1

	def model1(self, size):
		inpt = tf.placeholder(tf.float32, [size, self.sizeOfTupleInput])
		ans = tf.placeholder(tf.float32, [size])

		W1 = tf.Variable(tf.random_normal([self.sizeOfTupleInput, self.hiddenSz1], stddev=.1))
		b1 = tf.Variable(tf.random_normal([self.hiddenSz1], stddev=.1))
		l1Out = tf.matmul(inpt, W1) + b1
		L1Out = tf.nn.relu(l1Out)
		L1OutDrop = tf.nn.dropout(L1Out, self.keepPrb1)
		W2 = tf.Variable(tf.random_normal([self.hiddenSz1, self.outputSz], stddev=.1))
		b2 = tf.Variable(tf.random_normal([self.outputSz], stddev=.1))
		out = tf.nn.sigmoid(tf.matmul(L1OutDrop,W2)+b2)

		#loss corresponds to euclid dist
		loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(ans, out)), 1)) 
		sgd = tf.train.AdamOptimizer(self.learning_rate)
		train_op = sgd.minimize(loss)
		return [inpt, ans, W1, b1, l1Out, L1Out, L1OutDrop, W2, b2, out, loss, sgd, train_op]

