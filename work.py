import re
import seaborn as sns; sns.set()
import streamlit as st
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
from keras.models import load_model
import numpy as np
from nltk.corpus import stopwords  # module common words Natural Language (as,is ,and)
from nltk.stem import WordNetLemmatizer
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
stop_words = set(stopwords.words('english'))


def clean(text):
    # Lowering letters
    text = text.lower()
    # Removing html tags
    text = re.sub('<[^>]*>', '', text)
    # Removing emails
    text = re.sub('\S*@\S*\s?', '', text)
    # Removing urls
    text = re.sub('https?://[A-Za-z0-9]', '', text)
    # Removing numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)

    # Joining words
    text = (' '.join(filtered_sentence))
    return text



def predictSpamOrHam(text):
    cleaned_string = clean(text);
    tfidf = pickle.load(open('spam/vectorizer.pkl','rb'))
    model = pickle.load(open('spam/model.pkl','rb'))
    vector_input = tfidf.transform([cleaned_string])
    result = model.predict(vector_input)[0];
    if result == 0:
        return "Spam"
    else:
        return "Ham"

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # This line converts the text to lowercase
    text = " ".join([word for word in text.split() if word not in stop_words])  # this line splits the text into individual words then iterates over each word if the word is not present
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # This line splits the preprocessed text into individual words then then applies lemmatization to each word
    return text

def make_prediction(text):
    loaded_model = load_model('emotion/emotion_classifier.h5')
    print(text)
    tokenizer = pickle.load(open('emotion/tokenizer.pk1', 'rb'))
    text = clean(text)  # converts the text to lowercase, removes stopwords, and performs lemmatization.
    test_sequence = tokenizer.texts_to_sequences([text])  # This line converts the preprocessed test text into a sequence of integers
    test_padded_sequence = pad_sequences(test_sequence, maxlen=36, padding='post')  # maximum sequence length
    prediction = loaded_model.predict(test_padded_sequence)[0]  # his line passes the padded test sequence through the trained model using the predict method. It returns a predicted probability distribution over the 8 emotion classes.
    emotion_labels = ["anger", "fear", "joy", "sadness"]
    predicted_emotion = emotion_labels[np.argmax(prediction)]  # retrieve the corresponding emotion label
    return predicted_emotion

def predictEmotion(text):
    x =make_prediction(text)
    return x;


def predictTag(text):
    maxlen=177;
    tokenizer = Tokenizer()
    model = tf.keras.models.load_model("tag/finalModelfinall.h5")
    labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
    test_seq = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=maxlen)
    test_preds = [labels[np.argmax(i)] for i in model.predict(test_seq)]
    for news, label in zip(text, test_preds):
       return label


def global_work():
    sample = pd.read_csv("file.csv")
    sample = sample.rename(columns={'0': 'mails'})
    sample = sample.drop('Unnamed: 0', axis=1)
    ok = 0;
    sample['tag'] = 0;
    sample['type'] = 0;
    sample['emotion'] =0;
    index = 0;
    st.snow()
    for row in sample['mails']:
        spam = predictSpamOrHam(row);
        tag = predictTag([row])
        emo = make_prediction(row)
        sample.at[index, 'type'] = spam;
        sample.at[index, 'tag'] = tag;
        sample.at[index, 'emotion'] = emo;
        index = index + 1
    pd.DataFrame(sample).to_csv("final.csv")


