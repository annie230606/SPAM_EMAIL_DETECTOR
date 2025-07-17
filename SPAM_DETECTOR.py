import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Load model
model = load_model("spam_classifier_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load config
with open("config.json", "r") as f:
    config = json.load(f)
    max_sequence_len = config["max_sequence_len"]

def predict_message(message, tokenizer, model, max_len):
    if not message.strip():
        print("Empty message provided.")
        return "Invalid Input", 0.0
    # Preprocess the input
    message = message.lower()
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    # Make prediction
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Spam" if prob > 0.4 else "Not Spam"
    
    return label, prob

# streamlit app
st.title("SPAM EMAIL DETECTOR ")
input_text=st.text_input("Enter the message")
input_text=input_text.lower()

if st.button("Detect if scam "):
    
    label,prob = predict_message(input_text,tokenizer, model, max_sequence_len)
    if prob>0.4 :
     st.error(f"Prediction: {label} (Confidence: {prob:.4f})")
    else:
     st.success(f"Prediction: {label} (Confidence: {prob:.4f})")

sentiment_mapping = ["one", "two", "three", "four", "five"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")