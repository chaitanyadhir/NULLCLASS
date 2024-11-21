import random
import numpy as np
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import pickle
import streamlit as st
from googletrans import Translator
from langdetect import detect

nltk.download('punkt')
nltk.download('wordnet')

lemmetizer = WordNetLemmatizer()
translator = Translator()

chat_df = pd.read_csv(
    r'C:\projects\internship_project\chatbot_data.csv',
    dtype={'categories': str, 'title': str, 'abstract': str},
    low_memory=False
)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for _, row in chat_df.iterrows():
    category = row['categories']
    pattern = row['title']
    response = row['abstract']
    word_list = nltk.word_tokenize(pattern)
    words.extend(word_list)
    documents.append((word_list, category))
    if category not in classes:
        classes.append(category)

words = [lemmetizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmetizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(trainX), np.array(trainY), epochs=250, batch_size=5, verbose=1)

model.save("chatbot_model.h5")

st.title("Multi-Language Chatbot")
st.write("Interact with the chatbot in multiple languages!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

lang_options = ['en', 'es', 'fr', 'de', 'hi']  # English, Spanish, French, German, Hindi
selected_lang = st.selectbox("Choose your language:", lang_options, index=0)

user_input = st.text_input("Type your message:")

if user_input:
    detected_lang = detect(user_input)
    if detected_lang != 'en':
        user_input_translated = translator.translate(user_input, src=detected_lang, dest='en').text
    else:
        user_input_translated = user_input

    st.session_state.chat_history.append(f"You ({selected_lang}): {user_input}")

    chatbot_response = f"Responding to '{user_input_translated}'"  # Placeholder logic
    if selected_lang != 'en':
        chatbot_response_translated = translator.translate(chatbot_response, src='en', dest=selected_lang).text
    else:
        chatbot_response_translated = chatbot_response

    st.session_state.chat_history.append(f"Bot ({selected_lang}): {chatbot_response_translated}")

if st.session_state.chat_history:
    st.text_area("Chat History", "\n".join(st.session_state.chat_history), height=400)
