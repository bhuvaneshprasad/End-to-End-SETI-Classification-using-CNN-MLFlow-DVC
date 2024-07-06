# streamlit_app/app.py

import streamlit as st
import requests
from PIL import Image

# Define FastAPI backend URL
backend_url = "http://localhost:7384"

def main():
    st.title("SETI Signals Classifier")

    # Example: Upload file and send POST request to FastAPI endpoint
    uploaded_file = st.file_uploader("Choose an image to predict...", type=["png"])
    st.markdown("You can get sample images from [here](https://github.com/bhuvaneshprasad/End-to-End-SETI-Classification-using-CNN-MLFlow-DVC/tree/main/assets/test_images) to predict.")
    if uploaded_file is not None:
        with st.spinner('Predicting...'):
            files = {"file": uploaded_file}
            response = requests.post(f"{backend_url}/predict", files=files)
            if response.status_code == 200:
                st.json(response.json())
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                st.error("Failed to predict")

if __name__ == "__main__":
    main()
