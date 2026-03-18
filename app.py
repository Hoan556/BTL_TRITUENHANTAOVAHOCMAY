import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# cấu hình trang
st.set_page_config(page_title="Nhận dạng động vật", page_icon="🐾")

st.title("🐾 Nhận dạng động vật bằng AI")

# load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_model.h5")
    return model

model = load_model()

# THỨ TỰ PHẢI GIỐNG LÚC TRAIN DATASET
class_names = [
    "Bird",
    "Cat",
    "Dog",
    "Elephant",
    "Horse",
    "Tiger"
]

uploaded_file = st.file_uploader("Tải ảnh động vật", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Ảnh đã tải", width=400)

    img_size = model.input_shape[1]

    img = image.resize((img_size, img_size))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔎 Dự đoán"):

        prediction = model.predict(img_array)

        index = np.argmax(prediction)

        result = class_names[index]

        confidence = float(np.max(prediction)) * 100

        st.success(f"Kết quả: {result}")
        st.info(f"Độ tin cậy: {confidence:.2f}%")