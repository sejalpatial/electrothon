import streamlit as st
import requests

st.title("AI Engine Fault Detection")

st.write("Upload an engine sound file to detect possible faults.")

uploaded_file = st.file_uploader("Upload engine audio (.wav)", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Detect Fault"):

        files = {"file": uploaded_file.getvalue()}

        response = requests.post(
            "http://127.0.0.1:8000/detect",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:

            data = response.json()

            st.success(f"Detected Fault: {data['fault']}")
            st.write("Explanation:")
            st.write(data["explanation"])

        else:
            st.error("Error detecting fault")
