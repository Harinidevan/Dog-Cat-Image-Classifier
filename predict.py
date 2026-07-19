import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("models/dog_cat_classifier.keras")

# Image path (change this to the image you want to test)
img_path = "test_image.jpg"

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

confidence = prediction[0][0]

if confidence >= 0.5:
    print(f"🐶 Dog ({confidence*100:.2f}% confidence)")
else:
    print(f"🐱 Cat ({(1-confidence)*100:.2f}% confidence)")