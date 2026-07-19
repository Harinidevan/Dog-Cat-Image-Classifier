import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Dog & Cat Classifier",
    page_icon="🐶",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("models/dog_cat_classifier.keras")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#F7F9FC;
}

.title{
    text-align:center;
    color:#0E6FFF;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:20px;
    margin-bottom:20px;
}

.card{
    background-color:#262730;
    padding:25px;
    border-radius:15px;
    text-align:center;
}

.result{
    font-size:35px;
    font-weight:bold;
    color:white;
}

.confidence{
    font-size:22px;
    color:white;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<div class='title'>🐶🐱 Dog & Cat Image Classifier</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning using MobileNetV2</div>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Upload Image
# -----------------------------
st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array, verbose=0)

    confidence = prediction[0][0]

    st.divider()

    st.subheader("Prediction")

    if confidence >= 0.5:

        st.markdown(f"""
        <div class='card'>
        <div class='result'>🐶 DOG</div>
        <br>
        <div class='confidence'>
        Confidence : {confidence*100:.2f}%
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence))

        st.success("The uploaded image is predicted as a Dog.")

        st.balloons()

    else:

        cat_confidence = 1 - confidence

        st.markdown(f"""
        <div class='card'>
        <div class='result'>🐱 CAT</div>
        <br>
        <div class='confidence'>
        Confidence : {cat_confidence*100:.2f}%
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(cat_confidence))

        st.success("The uploaded image is predicted as a Cat.")

        st.snow()

st.markdown("---")

st.markdown(
    "<div class='footer'>Developed using TensorFlow • MobileNetV2 • Streamlit</div>",
    unsafe_allow_html=True
)