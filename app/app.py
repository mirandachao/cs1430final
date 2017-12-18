from flask import Flask, render_template 
from flask_socketio import SocketIO


 #Create app and set template folder to this
app = Flask(__name__, static_folder='../build')
app.config['SECRET_KEY'] = 'secret!' #obviously change this to an environment variable if we put this in production
socketio = SocketIO(app)


def setup_tf():
	print(' * Setting up the Tensor Flow model')

"""
#Route I wrote just to test ajax requests
@app.route('/api/test')
def test():
	print("test route reached!")


@socketio.on('connect', namespace='/chat')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/chat')
def test_disconnect():
    print('Client disconnected')
"""


@socketio.on('connected')
def connected(message):
    print(" * SocketIO connection established by webapp!")


#Home route
@app.route('/')
def home():
    return render_template('demo.html')


if __name__ == '__main__':
	setup_tf() #We want to make sure the model is set up before we start the server
	socketio.run(app)
