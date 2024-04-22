from flask import Flask, render_template, request
import numpy as np
import cv2
import base64
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import layers 
from keras.applications.nasnet import  preprocess_input

app = Flask(__name__)

def get_model_classif_nasnet_1():  
    
    #epoch = 15 --- 96.31%
    
    inputs = Input((96, 96, 3))

    x1 = layers.Conv2D(32,3,padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(32,3,padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(32,3,padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
#     x1 = layers.MaxPool2D(2,2)(x1)  

    
    x1_s = layers.SeparableConv2D(32,3,padding='same')(inputs)
    x1_s = layers.BatchNormalization()(x1_s)
    x1_s = layers.Activation('relu')(x1_s)
    x1_s = layers.SeparableConv2D(32,3,padding='same')(x1_s)
    x1_s = layers.BatchNormalization()(x1_s)
    x1_s = layers.Activation('relu')(x1_s)    
    x1_s = layers.SeparableConv2D(32,3,padding='same')(x1_s)
    x1_s = layers.BatchNormalization()(x1_s)
    x1_s = layers.Activation('relu')(x1_s)
    concetenated_0 = layers.concatenate([x1,x1_s])

    x2 = layers.Conv2D(64,3,padding='same')(concetenated_0)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(64,3,padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(64,3,padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    residual_concetenated_0 = layers.Conv2D(64,1,strides=1,padding='same')(concetenated_0)
    x2 = layers.add([x2,residual_concetenated_0])
    concetenates_x2_x1_s = layers.concatenate([x2,x1_s])
    x2 = layers.MaxPool2D(2,2)(concetenates_x2_x1_s)
    
    

    x2_s = layers.SeparableConv2D(64,3,padding='same')(x2)
    x2_s = layers.BatchNormalization()(x2_s)
    x2_s = layers.Activation('relu')(x2_s)
    x2_s = layers.SeparableConv2D(64,3,padding='same')(x2_s)
    x2_s= layers.BatchNormalization()(x2_s)
    x2_s= layers.Activation('relu')(x2_s)
    x2_s = layers.SeparableConv2D(64,3,padding='same')(x2_s)
    x2_s= layers.BatchNormalization()(x2_s)
    x2_s= layers.Activation('relu')(x2_s)
    x2_s = layers.Conv2D(96,1,strides=1,padding='same')(x2_s)
    x2_s = layers.add([x2_s,x2]) 
    x2_s = layers.MaxPool2D(2,2)(x2_s)
    
    x3 = layers.Conv2D(128,3,padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Conv2D(128,3,padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Conv2D(128,3,padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    residual_x2 = layers.Conv2D(128,1,strides=1,padding='same')(x2)
    x3 = layers.add([residual_x2,x3]) 
    
    x3_x3 = layers.Conv2D(128,3,padding='same')(x3)
    x3_x3 = layers.BatchNormalization()(x3_x3)
    x3_x3 = layers.Activation('relu')(x3_x3)
    x3_x3 = layers.Conv2D(128,3,padding='same')(x3_x3)
    x3_x3 = layers.BatchNormalization()(x3_x3)
    x3_x3 = layers.Activation('relu')(x3_x3)
    x3_x3 = layers.Conv2D(128,3,padding='same')(x3_x3)
    x3_x3 = layers.BatchNormalization()(x3_x3)
    x3_x3 = layers.Activation('relu')(x3_x3)
    x3_x3 = layers.add([x3,x3_x3]) 
    x3_x3 = layers.MaxPool2D(2,2)(x3_x3)
    
    
    concetenated_1 = layers.concatenate([x3_x3,x2_s])
    x3_s = layers.SeparableConv2D(128,3,padding='same')(concetenated_1)
    x3_s = layers.BatchNormalization()(x3_s)
    x3_s = layers.Activation('relu')(x3_s)
    x3_s = layers.SeparableConv2D(128,3,padding='same')(x3_s)
    x3_s= layers.BatchNormalization()(x3_s)
    x3_s= layers.Activation('relu')(x3_s)
    x3_s = layers.SeparableConv2D(128,3,padding='same')(x3_s)
    x3_s= layers.BatchNormalization()(x3_s)
    x3_s= layers.Activation('relu')(x3_s)
    x3_s = layers.add([x3_s,x3_x3]) 
    x3_s = layers.MaxPool2D(2,2)(x3_s)
    
    x4 = layers.Conv2D(256,3,padding='same')(x3_x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)
    x4 = layers.Conv2D(256,3,padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)
    x4 = layers.Conv2D(256,3,padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)
    residual_x3 = layers.Conv2D(256,1,strides=1,padding='same')(x3_x3)
    x4 = layers.add([residual_x3,x4]) 
    
    x4_x4 = layers.Conv2D(256,3,padding='same')(x4)
    x4_x4 = layers.BatchNormalization()(x4_x4)
    x4_x4 = layers.Activation('relu')(x4_x4)
    x4_x4 = layers.Conv2D(256,3,padding='same')(x4_x4)
    x4_x4 = layers.BatchNormalization()(x4_x4)
    x4_x4 = layers.Activation('relu')(x4_x4)
    x4_x4 = layers.Conv2D(256,3,padding='same')(x4_x4)
    x4_x4 = layers.BatchNormalization()(x4_x4)
    x4_x4 = layers.Activation('relu')(x4_x4)
    x4_x4 = layers.add([x4,x4_x4]) 
    x4_x4 = layers.MaxPool2D(2,2)(x4_x4)
    

    concetenated_2 = layers.concatenate([x4_x4,x3_s])
    x4_s = layers.SeparableConv2D(256,3,padding='same')(concetenated_2)
    x4_s = layers.BatchNormalization()(x4_s)
    x4_s = layers.Activation('relu')(x4_s)
    x4_s = layers.SeparableConv2D(256,3,padding='same')(x4_s)
    x4_s= layers.BatchNormalization()(x4_s)
    x4_s= layers.Activation('relu')(x4_s)
    x4_s = layers.SeparableConv2D(256,3,padding='same')(x4_s)
    x4_s= layers.BatchNormalization()(x4_s)
    x4_s= layers.Activation('relu')(x4_s)
    x4_s = layers.add([x4_s,x4_x4]) 
    x4_s = layers.MaxPool2D(2,2)(x4_s)
    
    x5 = layers.Conv2D(512,3,padding='same')(x4_x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Activation('relu')(x5)
    x5 = layers.Conv2D(512,3,padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Activation('relu')(x5)
    x5 = layers.Conv2D(512,3,padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Activation('relu')(x5)
    residual_x4 = layers.Conv2D(512,1,strides=1,padding='same')(x4_x4)
    x5 = layers.add([residual_x4,x5])

    x5_x5 = layers.Conv2D(512,3,padding='same')(x5)
    x5_x5 = layers.BatchNormalization()(x5_x5)
    x5_x5 = layers.Activation('relu')(x5_x5)
    x5_x5 = layers.Conv2D(512,3,padding='same')(x5_x5)
    x5_x5 = layers.BatchNormalization()(x5_x5)
    x5_x5 = layers.Activation('relu')(x5_x5)
    x5_x5 = layers.Conv2D(512,3,padding='same')(x5_x5)
    x5_x5 = layers.BatchNormalization()(x5_x5)
    x5_x5 = layers.Activation('relu')(x5_x5)
    x5_x5 = layers.add([x5,x5_x5])
    x5_x5 = layers.MaxPool2D(2,2)(x5_x5)
    
    concetenated_3 = layers.concatenate([x5_x5,x4_s])
    x5_s = layers.SeparableConv2D(512,3,padding='same')(concetenated_3)
    x5_s = layers.BatchNormalization()(x5_s)
    x5_s = layers.Activation('relu')(x5_s)
    x5_s = layers.SeparableConv2D(512,3,padding='same')(x5_s)
    x5_s= layers.BatchNormalization()(x5_s)
    x5_s= layers.Activation('relu')(x5_s)
    x5_s = layers.SeparableConv2D(512,3,padding='same')(x5_s)
    x5_s= layers.BatchNormalization()(x5_s)
    x5_s= layers.Activation('relu')(x5_s)
    x5_s = layers.add([x5_s,x5_x5]) 

    x = layers.GlobalAveragePooling2D()(x5_s)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    output_tensor = layers.Dense(1,activation='sigmoid')(x)

    model = Model(inputs,output_tensor)
    
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
    # model.summary()
    return model


# Load the model
def load_cancer_model():
    # Recreate the model architecture
    model = get_model_classif_nasnet_1()
    # Load the weights
    model.load_weights('model.h5')
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_cancer_model()


# Preprocess input image
def preprocess_image(image):
    # Your preprocessing logic here
    # Make sure it matches preprocessing used during training
    return preprocess_input(image)

@app.route('/')
def home():
    model_name='CNN'
    result = ''
    return render_template('index.html', result=result, model_name=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        prediction = model.predict(np.array([preprocessed_image]))
        
        # Process prediction as required
        result = "{:.2f}".format(prediction[0][0])  # Assuming prediction is a single value
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
