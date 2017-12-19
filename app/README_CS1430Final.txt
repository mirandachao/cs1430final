To run the webapp

0. Make sure to build: cd ../build && ./build_library
1. npm install, pip install flask, pip install flask-socketio, npm install tensorflow (I think)
2. python3 app.py
3. open localhost:5000 in web browser


General notes
- Flask server
- Serves the demo.html webpage
- Also sets up the tensor flow


Implementation Details
- Uses SocketIO to establish a bidirectional websockets connection
- client side js sends data (input) to server which outputs predicted screen coordinates using neural net

Read more:
- https://flask-socketio.readthedocs.io/en/latest/
- 