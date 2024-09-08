from flask import Flask, render_template, url_for, redirect, request
import tensorflow as tf
import tf_keras as keras
from PIL import Image
import io
from helper_functions import class_names

tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

model = keras.models.load_model("model")

@app.route("/")
def initial_page():
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("index.html",class_names=class_names)

@app.route("/predict",methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("index.html",class_names = class_names)

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html",class_names = class_names)
    
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = tf.image.resize(img,[224,224])
        img = tf.expand_dims(img,axis = 0)
        prediction = model.predict(img)
        probability = round(prediction.max(),3)
        prediction = class_names[prediction.argmax()] 
    return render_template("index.html",prediction = prediction,probability = probability,class_names = class_names)

if __name__ == "__main__":
    app.run(debug=True)
