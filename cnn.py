import pathlib
import base64
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from flask import Flask, render_template, request
import numpy as np
import cv2
from fastai.vision.all import *

app = Flask(__name__)

# Load the fastai model
def load_fastai_model():
    model = load_learner("model.pkl", cpu=True)
    return model

model = load_fastai_model()

# Preprocess input image
def preprocess_image(image):
    # Your preprocessing logic here
    # Make sure it matches preprocessing used during training
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return PILImage.create(image)

@app.route('/')
def home():
    model_name='DenseNet'
    result = ''
    return render_template('index.html', result=result,model_name=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
            # Get user details from the form
        name = request.form['name']
        age = request.form['age']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        # Get the image file from the request
        file = request.files['file']
        
        # Read the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Make prediction
        prediction, _, _ = model.predict(preprocessed_image)
        
        # Process prediction as required
        result = "{:.2f}".format(float(prediction))  # Format the string prediction

        _, img_encoded = cv2.imencode('.jpg', image)
        image_data = base64.b64encode(img_encoded).decode()
        # Debugging: Print prediction result
        print("Prediction:", result)
        
        return render_template('result.html', result=result, name=name, age=age, email=email, phone=phone, address=address,image_data=image_data)

    except Exception as e:
        # Error handling: Print error message
        print("Error occurred:", e)
        # Return an error message or redirect to home page
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
