import numpy as np # It provides efficient data structures for handling large arrays and matrices
import pandas as pd # used for data manipulation(tabular data)
from nltk.corpus import stopwords # module common words Natural Language (as,is ,and)
from nltk.stem import WordNetLemmatizer #module provides functionality for lemmatizing words in natural language processing(dogs to dog).
import gensim.downloader as api # which are useful for various NLP tasks like text classification, sentiment analysis
from keras.preprocessing.text import Tokenizer #used to ensure that all input sequences have the same length.
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout,Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import nltk
nltk.download('omw-1.4')

#import the libraries and download the necessary data:
nltk.download('stopwords')#This line downloads the NLTK stopwords(and ,is ,a)
nltk.download('wordnet')#This line downloads the WordNet , which is a lexical database for the English (nouns, verbs, adjectives and adverbs )

stop_words = stopwords.words('english') # used retrieves the list of English stopwords from NLTK and remove move common words from text
lemmatizer = WordNetLemmatizer()#helpful for text normalization and improving text analysis.create object lemmatizer
glove_vectors = api.load("glove-wiki-gigaword-300")

def get_embedding(word):
    try:
        return glove_vectors[word] #returns the corresponding pre-trained GloVe word embedding vector for that word.
    except KeyError:
        return None # is not present in the glove_vectors object

#define your functions for text preprocessing:
def preprocess_text(text):
    text = text.lower()#This line converts the text to lowercase
    text = " ".join([word for word in text.split() if word not in stop_words]) #this line splits the text into individual words then iterates over each word if the word is not present
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])#This line splits the preprocessed text into individual words then then applies lemmatization to each word
    return text

    # load and preprocess your data:


data = pd.read_csv(
    r"C:\Users\DELL\Downloads\archive (5)\eng_dataset.csv")  # his line reads a CSV file called "emotions.csv" into a Pandas DataFrame named data. The CSV file presumably contains text data and corresponding emotion labels.

data['sentiment'].loc[data['sentiment']=='anger']=0.0
data['sentiment'].loc[data['sentiment']=='fear']=1.0
data['sentiment'].loc[data['sentiment']=='joy']=2.0
data['sentiment'].loc[data['sentiment']=='sadness']=3.0
data['sentiment']=np.asanyarray(data['sentiment']).astype('float64')
data["contnet"] = data["content"].apply(preprocess_text)#It preprocesses the text data by converting it to lowercase, removing stopwords, and performing lemmatization
x_train,x_test,y_train,y_test=train_test_split(data['content'],
                                               data['sentiment'],
                                              test_size=0.2,
                                               random_state=2,
                                              )
tokenizer = Tokenizer()

tokenizer.fit_on_texts(x_train)
x_train_sequences = tokenizer.texts_to_sequences(x_train)
max_len = max([len(seq) for seq in x_train_sequences])#This line calculates the maximum length of the sequences,
x_train_padded = pad_sequences(x_train_sequences, maxlen=max_len, padding='post')

x_test_sequences=tokenizer.texts_to_sequences(x_test)
x_test_padded = pad_sequences(x_test_sequences,maxlen=max_len, padding='post')
word_index = tokenizer.word_index

embedding_dim = 300 #This line sets the dimensionality of the word embeddings to 100.
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = get_embedding(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#define and train your model:

model1 = Sequential([
    Embedding(len(word_index) + 1, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
])
model1.save('emotion_classifier1.h5')

#define and train your model:

model2 = Sequential([
    Embedding(len(word_index) + 1, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False),
    LSTM(32),
    Dense(32, activation='linear'),
    Activation('softmax')
])

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model2.fit(x_train_padded, y_train, epochs=30, validation_data=(x_test_padded,y_test))
model2.save('emotion_classifier2.h5')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

#Model 1

y_prediction=model1.predict(x_test_padded)

emotion_labels = ["anger", "fear",  "joy", "sadness"]
y_test.loc[y_test==0.0]='anger'
y_test.loc[y_test==1.0]='fear'
y_test.loc[y_test==2.0]='joy'
y_test.loc[y_test==3.0]='sadness'
# y_prediction_emotion = emotion_labels[np.argmax(y_prediction)]
y_prediction_emotion=[]
for y in y_prediction:
    prediction=emotion_labels[np.argmax(y)]
    y_prediction_emotion.append(prediction)
cm=confusion_matrix(y_test,y_prediction_emotion)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=emotion_labels)
disp.plot()
plt.show()
print("Model 1")
print("Accuracy Score "+str(accuracy_score(y_test,y_prediction_emotion)))

print("Percision Score"+str(precision_score(y_test,y_prediction_emotion,
                                            labels=emotion_labels,
                                            average=None
                                            )))
print("Recall Score"+str(recall_score(y_test,y_prediction_emotion,
                                            labels=emotion_labels,
                                            average=None
                                            )))
print("F1 Score"+str(f1_score(y_test,y_prediction_emotion,
                                            labels=emotion_labels,
                                            average=None
                                            )))
#Model 2

y_prediction=model2.predict(x_test_padded)

emotion_labels = ["anger", "fear",  "joy", "sadness"]
y_test.loc[y_test==0.0]='anger'
y_test.loc[y_test==1.0]='fear'
y_test.loc[y_test==2.0]='joy'
y_test.loc[y_test==3.0]='sadness'
# y_prediction_emotion = emotion_labels[np.argmax(y_prediction)]
y_prediction_emotion=[]
for y in y_prediction:
    prediction=emotion_labels[np.argmax(y)]
    y_prediction_emotion.append(prediction)
cm=confusion_matrix(y_test,y_prediction_emotion)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=emotion_labels)
disp.plot()
plt.show()
print("Model 2")
print("Accuracy Score "+str(accuracy_score(y_test,y_prediction_emotion)))

print("Percision Score"+str(precision_score(y_test,y_prediction_emotion,
                                            labels=emotion_labels,
                                            average=None
                                            )))
print("Recall Score"+str(recall_score(y_test,y_prediction_emotion,
                                            labels=emotion_labels,
                                            average=None
                                            )))
print("F1 Score"+str(f1_score(y_test,y_prediction_emotion,
                                            labels=emotion_labels,
                                            average=None
                                            )))



from keras.models import load_model
loaded_model = load_model('emotion_classifier1.h5')
loaded_model.summary()

def make_prediction(text):
    text = preprocess_text(text)# converts the text to lowercase, removes stopwords, and performs lemmatization.
    test_sequence = tokenizer.texts_to_sequences([text])#This line converts the preprocessed test text into a sequence of integers
    test_padded_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post')#maximum sequence length
    prediction = loaded_model.predict(test_padded_sequence)[0]#his line passes the padded test sequence through the trained model using the predict method. It returns a predicted probability distribution over the 8 emotion classes.
    emotion_labels = ["anger", "fear",  "joy", "sadness"]
    predicted_emotion = emotion_labels[np.argmax(prediction)]#retrieve the corresponding emotion label
    print(predicted_emotion)

make_prediction('grateful, lucky and thankful')
make_prediction('I feel so happy today')
make_prediction('I cried when I won the medal')
make_prediction('I cried when I saw the accident')
make_prediction('Get out!!!')
make_prediction('He left the room and kicked the door')
make_prediction('He shouted with wrath')
make_prediction('I shouted when I was riding the roller coaster')

