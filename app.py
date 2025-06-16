import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from gradcam import show_gradcam

model = load_model("model_cnn.h5")
class_labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

st.title("🦵 Knee Arthritis Detector")
st.write("Upload a knee X-ray to detect osteoarthritis severity.")

uploaded_file = st.file_uploader("Choose an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_path = f"temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    st.success(f"Predicted KL Grade: **{class_labels[class_idx]}**")

    st.subheader("Grad-CAM Heatmap")
    show_gradcam(img_path, class_labels=class_labels)
