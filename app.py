# app.py
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model # type: ignore

from tensorflow.keras.preprocessing import image # type: ignore

import numpy as np

app = Flask(__name__)
model = load_model('model/skin_cancer_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route: Prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)[0][0]
    prediction = "Malignant" if result > 0.5 else "Benign"

    return render_template('result.html', prediction=prediction, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
