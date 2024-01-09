import re

import gensim.downloader as api
import nltk
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("punkt")

physical_devices = tf.config.get_visible_devices("GPU")
print(tf.config.list_physical_devices())
if not physical_devices:
    print("No GPU available. Switching to CPU.")
else:
    print("GPU available. Using GPU.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Function to preprocess sentences
def preprocess_sentence(sentence):
    # Remove non-alphabetic characters and convert to lowercase
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    sentence = sentence.lower().split()

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    sentence = [word for word in sentence if word not in stop_words]

    return sentence


# Load IMDb dataset
num_words = None
skip_top = 0
maxlen = None
seed = 113
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=num_words, skip_top=skip_top, maxlen=maxlen, seed=seed
)

# Combine the training and testing data
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# Convert integer sequences to text
word_index = imdb.get_word_index()
index_to_word = {i: word for word, i in word_index.items()}
X_sentences = [
    " ".join([index_to_word.get(word - 3, "") for word in review]) for review in X
]

# Tokenize sentences
X_tokenized = [sent_tokenize(review) for review in X_sentences]

word2vec_model = api.load("word2vec-google-news-300")


# Function to get sentence embeddings
def get_sentence_embeddings(sentences, model):
    embeddings = []
    for sentence in sentences:
        preprocessed_sentence = preprocess_sentence(sentence)
        valid_words = [
            word for word in preprocessed_sentence if word in model.key_to_index
        ]
        if valid_words:
            embeddings.append(np.mean(model[valid_words], axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)


# Get sentence embeddings
X_embeddings = get_sentence_embeddings(X_sentences, word2vec_model)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42
)

from sklearn.neighbors import KNeighborsClassifier

# Train the classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
