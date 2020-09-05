""" Code file to run LSTM model and get predictions """

import re
import pandas as pd
import numpy as np
from typing import Dict, KeysView, Set, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.noise import GaussianNoise
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


def cutter(word: str, wnl: WordNetLemmatizer) -> str:
    """ Lemmatizes a given word to its verb form
        Args:
            word: Input word to lemmatize
            wnl: Word lemmatizer from nltk wordnet
        Returns:
            Lemmatized word
    """
    if len(word) < 4:
        return word

    return wnl.lemmatize(wnl.lemmatize(word, "n"), "v")


def pre_process_all(string: str, wnl: WordNetLemmatizer) -> str:
    """ Pre process question strings for abbreviations and numbers and lemmatize individual words
        Args:
            string: Input question to pre process
            wnl: Word lemmatizer from nltk wordnet
        Returns:
            Pre processed question string
    """
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    string = ' '.join([cutter(w, wnl) for w in string.split()])

    return string


def get_embedding(embedding_file: str, embedding_dim: int, top_words: Set[str]) -> Dict[str, np.array]:
    """ Creates and embedding dictionary using Glove embedding and the frequent words in the vocabulary
        Args:
            embedding_file: Embedding file name (Glove Embedding chosen here)
            embedding_dim: Chosen embedding length
            top_words: Frequent words in the vocabulary from all questions
    """
    embeddings_index = {}
    f = open(embedding_file, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) == embedding_dim + 1 and word in top_words:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def is_numeric(s: str) -> bool:
    """ Check if there are digits in a word
        Args:
            s: Input word
        Return:
            True if digits are present in the word, False otherwise
    """
    return any(i.isdigit() for i in s)


def prepare(q: str, top_words: KeysView[str], stop_words: Set[str],
            max_sequence_length: int) -> Tuple[str, Set[str], Set[str]]:
    """ Prepare a question string for modelling by discarding stop-words and replacing
        non common words by a single word - "memento. Restrict length to max sequence length
        Args:
            q: Input question string
            top_words: List of common words to be used in modelling
            stop_words: Set of stop words to be discarded
            max_sequence_length: Maximum length of a word that is to be preserved
        Returns:
            A tuple of processed question string, set of words with digits and set of surplus words
    """
    new_q = []
    surplus_q = []
    numbers_q = []
    new_memento = True
    for w in q.split()[::-1]:
        if w in top_words:
            new_q = [w] + new_q
            new_memento = True
        elif w not in stop_words:
            if new_memento:
                new_q = ["memento"] + new_q
                new_memento = False
            if is_numeric(w):
                numbers_q = [w] + numbers_q
            else:
                surplus_q = [w] + surplus_q
        else:
            new_memento = True
        if len(new_q) == max_sequence_length:
            break
    new_q = " ".join(new_q)

    return new_q, set(surplus_q), set(numbers_q)


def extract_features(df: pd.DataFrame, top_words: KeysView[str],
                     stop_words: Set[str],
                     max_sequence_length: int) -> Tuple[np.array, np.array, np.array]:
    """ Process question strings and create features from them to be used in modelling
        Args:
            df: Train or test data frame
            top_words: List of common words to be used in modelling
            stop_words: Set of stop words to be discarded
            max_sequence_length: Maximum length of a word that is to be preserved
        Returns:
            A tuple of processed question strings and features

    """
    q1s = np.array([""] * len(df), dtype=object)
    q2s = np.array([""] * len(df), dtype=object)
    features = np.zeros((len(df), 4))

    for i, (q1, q2) in enumerate(list(zip(df["question1"], df["question2"]))):
        q1s[i], surplus1, numbers1 = prepare(q1, top_words, stop_words, max_sequence_length)
        q2s[i], surplus2, numbers2 = prepare(q2, top_words, stop_words, max_sequence_length)
        features[i, 0] = len(surplus1.intersection(surplus2))
        features[i, 1] = len(surplus1.union(surplus2))
        features[i, 2] = len(numbers1.intersection(numbers2))
        features[i, 3] = len(numbers1.union(numbers2))

    return q1s, q2s, features


def run_lstm_with_embeddings(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """ Runs model using LSTM and Glove Embeddings
        Args:
            train: DataFrame for Training data
            test: DataFrame for Test data
        Returns:
            Does not return a value.
            Saves individual model prediction csv files instead in the predictions folder
    """
    np.random.seed(0)
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    max_sequence_length = 30
    min_word_occurrence = 100
    replace_word = "memento"
    embedding_dim = 300
    num_folds = 10
    batch_size = 1025
    embedding_file = "glove.840B.300d.txt"

    train["question1"] = train["question1"].fillna("").apply(pre_process_all, wnl)
    train["question2"] = train["question2"].fillna("").apply(pre_process_all, wnl)

    print("Creating the vocabulary of words occurred more than", min_word_occurrence)
    all_questions = pd.Series(train["question1"].tolist() + train["question2"].tolist()).unique()
    vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=min_word_occurrence)
    vectorizer.fit(all_questions)
    top_words = set(vectorizer.vocabulary_.keys())
    top_words.add(replace_word)

    embeddings_index = get_embedding(embedding_file, embedding_dim, top_words)
    print("Words are not found in the embedding:", top_words - embeddings_index.keys())
    top_words = embeddings_index.keys()

    print("Train questions are being prepared for LSTM...")
    q1s_train, q2s_train, train_q_features = \
        extract_features(train, top_words, stop_words, max_sequence_length)

    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(np.append(q1s_train, q2s_train))
    word_index = tokenizer.word_index

    data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), maxlen=max_sequence_length)
    data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), maxlen=max_sequence_length)
    labels = np.array(train["is_duplicate"])

    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("Train features are being merged with NLP and Non-NLP features...")
    train_nlp_features = pd.read_csv("data/nlp_features_train.csv")
    train_non_nlp_features = pd.read_csv("data/non_nlp_features_train.csv")
    features_train = np.hstack((train_q_features, train_nlp_features, train_non_nlp_features))

    print("Same steps are being applied for test...")
    test["question1"] = test["question1"].fillna("").apply(pre_process_all, args=(wnl,))
    test["question2"] = test["question2"].fillna("").apply(pre_process_all, args=(wnl,))
    q1s_test, q2s_test, test_q_features = \
        extract_features(test, top_words, stop_words, max_sequence_length)
    test_data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_test), maxlen=max_sequence_length)
    test_data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_test), maxlen=max_sequence_length)
    test_nlp_features = pd.read_csv("data/nlp_features_test.csv")
    test_non_nlp_features = pd.read_csv("data/non_nlp_features_test.csv")
    features_test = np.hstack((test_q_features, test_nlp_features, test_non_nlp_features))

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    model_count = 0

    for idx_train, idx_val in skf.split(train["is_duplicate"], train["is_duplicate"]):
        print("MODEL:", model_count)
        data_1_train = data_1[idx_train]
        data_2_train = data_2[idx_train]
        labels_train = labels[idx_train]
        f_train = features_train[idx_train]

        data_1_val = data_1[idx_val]
        data_2_val = data_2[idx_val]
        labels_val = labels[idx_val]
        f_val = features_train[idx_val]

        embedding_layer = Embedding(nb_words,
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=False)
        lstm_layer = LSTM(75, recurrent_dropout=0.2)

        sequence_1_input = Input(shape=(max_sequence_length,), dtype="int32")
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        sequence_2_input = Input(shape=(max_sequence_length,), dtype="int32")
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer(embedded_sequences_2)

        features_input = Input(shape=(f_train.shape[1],), dtype="float32")
        features_dense = BatchNormalization()(features_input)
        features_dense = Dense(200, activation="relu")(features_dense)
        features_dense = Dropout(0.2)(features_dense)

        addition = add([x1, y1])
        minus_y1 = Lambda(lambda x: -x)(y1)
        merged = add([x1, minus_y1])
        merged = multiply([merged, merged])
        merged = concatenate([merged, addition])
        merged = Dropout(0.4)(merged)

        merged = concatenate([merged, features_dense])
        merged = BatchNormalization()(merged)
        merged = GaussianNoise(0.1)(merged)

        merged = Dense(150, activation="relu")(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)

        out = Dense(1, activation="sigmoid")(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=out)
        model.compile(loss="binary_crossentropy",
                      optimizer="nadam")
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        best_model_path = "best_model" + str(model_count) + ".h5"
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

        hist = model.fit([data_1_train, data_2_train, f_train], labels_train,
                         validation_data=([data_1_val, data_2_val, f_val], labels_val),
                         epochs=15, batch_size=batch_size, shuffle=True,
                         callbacks=[early_stopping, model_checkpoint], verbose=1)

        model.load_weights(best_model_path)
        print(model_count, "validation loss:", min(hist.history["val_loss"]))

        predictions = model.predict([test_data_1, test_data_2, features_test],
                                    batch_size=batch_size, verbose=1)

        submission = pd.DataFrame({"test_id": test["test_id"], "is_duplicate": predictions.ravel()})
        submission.to_csv("predictions/preds" + str(model_count) + ".csv", index=False)

        model_count += 1
