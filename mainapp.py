import json
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import re
import random
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('chatbot_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as handle:
    lbl_enc = pickle.load(handle)

# Load the DataFrame
with open('intents.json', 'r') as file:
    data = json.load(file)

# Extract the intents
intents = data["intents"]  

# Function to preprocess input text
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z\']', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = " ".join(text.split())
    return text

# Function to get model response
def get_model_response(query, intents):
    preprocessed_query = preprocess_text(query)
    
    # Tokenize the input
    x_test = tokenizer.texts_to_sequences([preprocessed_query])
    
    # Check if the tokenized input is empty
    if not x_test[0]:  # If empty
        return "I'm sorry, I couldn't understand that."
    
    # Pad sequences to ensure consistent length
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=18)
    
    # Get model prediction
    y_pred = model.predict(x_test)
    predicted_class = np.argmax(y_pred)
    tag = lbl_enc.inverse_transform([predicted_class])[0]
    
    # Get responses from intents
    responses = [intent["responses"] for intent in intents if intent["tag"] == tag]
    if responses:
        bot_response = random.choice(responses[0])
    else:
        bot_response = "I'm sorry, I couldn't understand that."
    
    return bot_response

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Streamlit app
def main():
    st.title("Mental Health Chatbot")
    
    # Initialize session state
    session_state = SessionState(conversation=[], user_input="")
    
    # Display conversation history in chat-like format
    for message in session_state.conversation:
        if message.startswith("You:"):
            st.text_input("You:", message[5:], disabled=True)
        elif message.startswith("Bot:"):
            st.text_input("Bot:", message[5:], disabled=True)
    
    # Get user input
    user_input = st.text_input("You:", session_state.user_input, key='user_input')
    
    # Process user input and get bot response
    if st.button("Send"):
        if user_input.strip() == "":
            st.error("Please enter a message.")
        else:
            bot_response = get_model_response(user_input, intents)
            session_state.conversation.append(f"You: {user_input}")
            session_state.conversation.append(f"Bot: {bot_response}")
            st.text_input("Bot:", bot_response, disabled=True)

if __name__ == "__main__":
    main()
