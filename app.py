
import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Emotion Detector")
st.title("Emotion Detection - Happy / Sad / Angry")
st.write("Enter any sentence to detect emotion")

text = st.text_area("Enter text here:")

if st.button("Detect Emotion"):
    if text:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        st.success(f"Predicted Emotion: {pred}")
    else:
        st.write("Please enter some text first 
        st.markdown("Made by Ekta & Arsh")
