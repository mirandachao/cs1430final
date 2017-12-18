from flask import Flask, render_template
 
 #Create app and set template folder to this
app = Flask(__name__, template_folder='./', static_folder='./build') 

@app.route('/')
def home():
    return render_template('demo.html')


if __name__ == '__main__':
    app.run(port=int("8000"))
