import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

@st.cache_resource
def load_decoder():
    return tf.keras.models.load_model("vae_decoder.h5")

decoder = load_decoder()

st.title("✍️ Handwritten Digit Generator (0–9)")

digit = st.selectbox("Choose a digit (0–9) for label control (optional)", list(range(10)))

if st.button("Generate 5 Digits"):
    st.write("Generated images:")
    cols = st.columns(5)
    for i in range(5):
        z_sample = np.random.normal(size=(1, 2))
        generated = decoder.predict(z_sample)
        img = generated.reshape(28, 28)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        cols[i].pyplot(fig)
