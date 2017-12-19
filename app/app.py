from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from random import randrange
import numpy as np
import time
import tensorflow as tf
import pprint

 #Create app and set template folder to this
app = Flask(__name__, static_folder='../build')
app.config['SECRET_KEY'] = 'secret!' #obviously change this to an environment variable if we put this in production
socketio = SocketIO(app)
graphs = {}
cnns = {}

class network:
	def model1(self, size):
		self.inpt = tf.placeholder(tf.float32, [size, self.rows, self.cols, self.channelsIn])
		self.ans = tf.placeholder(tf.float32, [size])
		self.pos = tf.placeholder(tf.float32, [size])

		wc1 = tf.Variable(tf.random_normal([self.conv1Sz, self.conv1Sz, self.channelsIn, self.conv1Channels], stddev=.1))
		conv1 = tf.nn.conv2d(self.inpt, wc1, [1,1,1,1], 'VALID') #con1 shape is [batchSz, 20, 20, channels]
		avg = tf.nn.pool(conv1, [self.avgPoolSz, self.avgPoolSz], 'AVG', 'VALID', strides=[self.strideSz, self.strideSz])
		newRowSize = int((self.rows-(self.conv1Sz-1))/self.avgPoolSz)
		newColSize = int((self.cols-(self.conv1Sz-1))/self.avgPoolSz)

		matMulShape = tf.reshape(avg, [size, newRowSize * newColSize * self.conv1Channels])

		W1 = tf.Variable(tf.random_normal([newRowSize * newColSize * self.conv1Channels, self.hiddenSz1], stddev=.1))
		b1 = tf.Variable(tf.random_normal([self.hiddenSz1], stddev=.1))
		l1Out = tf.matmul(matMulShape, W1) + b1
		W2 = tf.Variable(tf.random_normal([self.hiddenSz1, self.outputSz], stddev=.1))
		b2 = tf.Variable(tf.random_normal([self.outputSz], stddev=.1))
		val = tf.nn.sigmoid(tf.matmul(l1Out,W2)+b2)
		wVal = tf.Variable(tf.random_normal([size], stddev=.1))
		wPos = tf.Variable(tf.random_normal([size], stddev=.1))
		self.out = tf.multiply(wVal, val) + tf.multiply(wPos, self.pos)

		#loss corresponds to euclid dist
		loss = tf.reduce_sum(tf.square(self.ans-self.out))
		sgd = tf.train.AdamOptimizer(self.learning_rate)
		train_op = sgd.minimize(loss)
		return [self.inpt, self.ans, self.pos, wc1, conv1, avg, newRowSize, newColSize, matMulShape, \
			W1, b1, l1Out, W2, b2, val, wVal, wPos, self.out, loss, sgd, train_op]

	def setup_network(self, label):
		inpt, ans, pos, wc1, conv1, avg, newRowSize, newColSize, matMulShape, W1, b1, l1Out, W2, b2, val, wVal, wPos, out, loss, sgd, train_op = self.model1(1)

		saver = tf.train.Saver()

		saver.restore(self.sess, "./weights/"+label)

	def eval(self, inny, posy):
		outy = self.sess.run(self.out, feed_dict={self.inpt: inny, self.pos: posy, self.ans: [0]})
		return outy

	def __init__(self, label):
		self.hiddenSz1 = 16
		self.learning_rate = 0.03
		self.rows = 24
		self.cols = 24
		self.conv1Sz = 5
		self.conv1Channels = 8
		self.avgPoolSz = 4
		self.strideSz = 4
		self.channelsIn = 1
		self.outputSz = 1
		self.sess = tf.Session()
		self.setup_network(label)

def setup_tf():
	print(' * Setting up the Tensor Flow model')
	graphs = {"leftX" : tf.Graph(), "leftY" : tf.Graph(), "rightX" : tf.Graph(), "rightY" : tf.Graph()}

	with graphs["leftX"].as_default():
		cnns["leftX"] = network("leftX")
	with graphs["leftY"].as_default():
		cnns["leftY"] = network("leftY")
	with graphs["rightX"].as_default():
		cnns["rightX"] = network("rightX")
	with graphs["rightY"].as_default():
		cnns["rightY"] = network("rightY")
"""
#Route I wrote just to test socket requests
@app.route('/api/test')
def test():
	print("test route reached!")


@socketio.on('connect', namespace='/chat')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/chat')
def test_disconnect():
    print('Client disconnected')


@socketio.on('predict')
def predict(json):
    #print('received json: ' + str(json))
    return randrange(1, 300), randrange(1,300)

"""
@socketio.on('connected')
def connected(message):
    print(" * SocketIO connection established by webapp!")


@app.route('/api/predict', methods=['POST'])
def tf_predict():
	#print("Hit this route! ", request.get_json())
	eyeData = request.get_json()

	x = np.array(eyeData)
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(eyeData)
	pic = np.random.rand(1, 24, 24, 1)
	leftX = cnns["leftX"].eval(pic, [.5])
	leftY = cnns["leftY"].eval(pic, [.5])
	rightX = cnns["rightX"].eval(pic, [.5])
	rightY = cnns["rightY"].eval(pic, [.5])


	return jsonify({'x':randrange(1, 300), 'y':randrange(1, 300)})


#Home route
@app.route('/')
def home():
    return render_template('demo.html')


if __name__ == '__main__':
	setup_tf() #We want to make sure the model is set up before we start the server
	socketio.run(app)
