# Final Project - COMP377- AI FOR SOFTWARE DEVELOPERS
# GROUP 3
# Muhammad Hassan - 301178235
# Adrian Dumitriu - 300566849
# Ivan Zapanta - 301173877

from flask import Flask, render_template, send_from_directory, url_for, request
import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField, ValidationError
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'MY Secret'
app.config['UPLOADED_PHOTOS_DEST'] = 'images'

# validate the uploaded image
def is_image(form, field):
    if field.data and field.data.filename:
        filename = field.data.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in {'.png', '.jpg', '.jpeg', '.gif'}:
            raise ValidationError('Upload a valid photo')

# Upload form 
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileRequired('File field cannot be empty'), is_image])
    submit = SubmitField('Upload')

# Loading mnist digit data for training
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Use 10000 images for training
training_set_size = 10000
x_train = x_train[0:training_set_size].reshape(training_set_size, 28, 28, 1)
y_train = y_train[0:training_set_size]
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build the Conv2D model
model = keras.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation=tf.keras.activations.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# Fit the model
model.fit(x_train, y_train, epochs=5)

fileNames = []
fileNames2 = []
imNumbers = []

# Controller to get the uploaded image to display
@app.route("/images/<filename>")
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

# Process the image before predicting
def preprocess_image(image):
    # Convert to grayscale
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_im = cv2.threshold(gray_im, 100, 255, cv2.THRESH_BINARY)
    mean_value = np.mean(thresh_im)
    # invert the image to make background black if needed
    if mean_value >= 120:
        thresh_im = 255 - thresh_im
    contours, _ = cv2.findContours(thresh_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour to capture digit
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if(w<h/2):
        x=math.floor(x+w/2-h/2)
        w=h
    
    if(h<w/2):
        y=math.floor(y+h/2-w/2)
        h=w

    digit_im = thresh_im[y:y+h, x:x+w]
    digit_im = cv2.resize(digit_im, (18, 18), interpolation=cv2.INTER_AREA)

    # add padding
    padded_im = cv2.copyMakeBorder(digit_im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
    return padded_im.reshape(1, 28, 28, 1)

# Controller to serve the main page and upload image
@app.route("/", methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    success = False  # Add this line
    if form.validate_on_submit():
        photo = form.photo.data
        filename = secure_filename(photo.filename)
        photo.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        file_url = url_for('get_file', filename=filename)
        fileNames.append(file_url)
        im = cv2.imread("images/" + filename)

        preprocessed_im = preprocess_image(im)
        cv2.imwrite(app.config['UPLOADED_PHOTOS_DEST']+'/p_'+filename, preprocessed_im.reshape(28,28))
        filename2="p_"+filename
        file_tr = url_for('get_file', filename=filename2)
        fileNames2.append(file_tr)
        # Predict the digit
        num = model.predict(preprocessed_im)
        imNumbers.append(np.argmax(num, axis=1)[0])

        success = True  # Add this line
    else:
        file_url = None
    return render_template('predict.html', form=form, file_url=fileNames, num=imNumbers, file_url2=fileNames2, success=success)



# App default config to run the app
if __name__ == "__main__":
    app.run(debug=False)