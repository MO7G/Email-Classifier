import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Embedding

# Dataset: http://qwone.com/~jason/20Newsgroups/

print("########## TEXT PREPROCESSING ##########")


# Create a function to read files
# Don't forget to include slash at the end of the path
def read_files(path):
    file_contents = list()
    filenames = os.listdir(path)

    for i in range(len(filenames)):
        with open(path + filenames[i],encoding="utf8", errors='ignore') as f:
            file_contents.append(f.read())

    return file_contents


# Read all required files
class_0 = read_files('TrainingData/20news-18828/sci.crypt/')
class_1 = read_files('TrainingData/20news-18828/rec.sport.baseball/')
class_2 = read_files('TrainingData/20news-18828/sci.med/')
class_3 = read_files('TrainingData/20news-18828/talk.politics.misc/')

# Defining labels
labels = ['comp.graphics', 'rec.motorcycles', 'sci.med', 'talk.politics.misc']

# Put all texts to a list
all_texts = np.append(class_0, class_1)
all_texts = np.append(all_texts, class_2)
all_texts = np.append(all_texts, class_3)

# Importing stopwords list
stop_words = set(stopwords.words('english'))


# Define a function to clean a text
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


# Clean all texts
all_cleaned_texts = np.array([clean(text) for text in all_texts])

# Create word-to-number mapping
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_cleaned_texts)

# Encode all words into numbers
all_encoded_texts = tokenizer.texts_to_sequences(all_cleaned_texts)
all_encoded_texts = np.array(all_encoded_texts,dtype=object)

# Checking the length of the first 5 files
for i in range(5):
    print('Length of file', i, ':', len(all_encoded_texts[i]))

# Ensure that all files are having the exact same length. In this case it's 500 words
all_encoded_texts = sequence.pad_sequences(all_encoded_texts, maxlen=500)

print("########## LABEL PREPROCESSING ##########")
# Create labels based on the length of each class
labels_0 = np.array([0] * len(class_0))
labels_1 = np.array([1] * len(class_1))
labels_2 = np.array([2] * len(class_2))
labels_3 = np.array([3] * len(class_3))

# Put all labels into a single list
all_labels = np.append(labels_0, labels_1)
all_labels = np.append(all_labels, labels_2)
all_labels = np.append(all_labels, labels_3)

# Create a new axis (this is just the shape expected by OneHotEncoder())
all_labels = all_labels[:, np.newaxis]

# Convert labels into one-hot representation
one_hot_encoder = OneHotEncoder(sparse=False)
all_labels = one_hot_encoder.fit_transform(all_labels)

print("########## MODEL TRAINING ##########")
# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(all_encoded_texts, all_labels,
                                                    test_size=0.2, random_state=11)

# Construct the neural network
model = Sequential()
model.add(Embedding(input_dim=35362, output_dim=32, input_length=500))
model.add(LSTM(100))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=12, batch_size=64,
                    validation_data=(X_test, y_test))

# Save the model for future use
model.save('IHopeThisOneWork.h5')

# Display the training process
plt.figure(figsize=(9, 7))
plt.title('Accuracy score')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

plt.figure(figsize=(9, 7))
plt.title('Loss value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

print("########## MODEL EVALUATION ##########")

# Predict test data
predictions = model.predict(X_test)
predictions = one_hot_encoder.inverse_transform(predictions)

# Convert y_test (actual label) from one-hot format
# to be in the same form of predictions array
y_test_evaluate = np.argmax(y_test, axis=1)

# Construct a confusion matrix
cm = confusion_matrix(y_test_evaluate, predictions)

# Display the confusion matrix
plt.figure(figsize=(8, 8))
plt.title('Confusion matrix on test data')
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
            cmap=plt.cm.Blues, cbar=False, annot_kws={'size': 14})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Create a new string
string = 'I just purchased a new motorcycle, I feel like it is a lot better than cars'

# Clean the new string
cleaned_string = clean(string)

# Encode all words in the string
encoded_string = tokenizer.texts_to_sequences([cleaned_string])

# Add zero padding such that the string will be having the length of 500 words
encoded_string = sequence.pad_sequences(encoded_string, maxlen=500)

# Predict string class
string_predict = model.predict(encoded_string)
print(np.argmax(string_predict))