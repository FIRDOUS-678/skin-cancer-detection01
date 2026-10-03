
import json
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="centered")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open("model_info.json") as f:
        info = json.load(f)
    if info["type"] == "dl":
        import tensorflow as tf
        model = tf.keras.models.load_model(info["model_file"])
    else:
        import joblib
        model = joblib.load(info["model_file"])
    return info, model

def predict(info, model, img):
    img = img.convert("RGB")
    if info["type"] == "dl":
        s = info["img_size"]
        arr = np.array(img.resize((s, s), Image.BILINEAR), dtype=np.float32)[None, ...]
        return model.predict(arr, verbose=0)[0]
    from features import extract_features_pil
    feats = extract_features_pil(img)[None, ...]
    return model.predict_proba(feats)[0]

info, model = load_model()
classes = info["class_names"]

st.title("🔬 Skin Lesion Classifier")
st.caption(f"Model: **{info['name']}**  |  Test accuracy: **{info['test_metrics']['acc']:.1%}**")
st.warning("Educational demo only. This is NOT a medical device and must not be used for diagnosis. "
           "Please consult a qualified dermatologist about any skin concern.")

tab_up, tab_cam = st.tabs(["Upload image", "Use camera"])
with tab_up:
    file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
with tab_cam:
    cam = st.camera_input("Take a photo")
src = file or cam

if src is not None:
    image = Image.open(src)
    st.image(image, caption="Input image", use_container_width=True)
    with st.spinner("Analysing..."):
        probs = predict(info, model, image)
    top = int(np.argmax(probs))
    st.subheader(f"Prediction: {classes[top]}")
    st.metric("Confidence", f"{probs[top]:.1%}")
    st.bar_chart({c: float(p) for c, p in zip(classes, probs)})
