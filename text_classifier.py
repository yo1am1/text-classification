import gensim.downloader as api
import joblib
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

physical_devices = tf.config.get_visible_devices("GPU")
if not physical_devices:
    print("No GPU available. Switching to CPU.")
else:
    print("GPU available. Using GPU.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

word2vec_model = api.load("word2vec-google-news-300")


def get_sentence_embeddings(sentences, model):
    embeddings = []
    for sentence in sentences:
        valid_words = [word for word in sentence if word in model.key_to_index]
        if valid_words:
            embeddings.append(np.mean(model[valid_words], axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)


X_embeddings = get_sentence_embeddings(X, word2vec_model)

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42
)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

joblib.dump(classifier, "word2vec_classifier.joblib")
word2vec_model.save("word2vec_model.bin")
