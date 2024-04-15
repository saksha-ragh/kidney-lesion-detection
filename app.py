from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained VGG16 model
model_path = 'C:\Users\Admin\Desktop\Kidney_lesion_detection\model.py'
model = load_model(model_path)

# Define class names for predictions
class_names = ['Normal', 'Cyst', 'Tumor', 'Stone']

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file uploaded.')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction='No selected file.')

    if file:
    
        img_path = 'uploads/' + file.filename
        file.save(img_path)

        # Preprocess the image
        img_array = preprocess_image(img_path)

        # Make prediction
        prediction = VGG_model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Delete the uploaded image
        os.remove(img_path)

        return render_template('index.html', prediction=f'Predicted class: {predicted_class}')


if __name__ == '__main__':
    app.run(debug=True)
