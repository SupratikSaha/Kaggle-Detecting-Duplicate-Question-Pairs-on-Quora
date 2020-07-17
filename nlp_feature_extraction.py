""" Code file to extract nlp features from the questions """

import re
import pandas as pd
from typing import List, Tuple, Set
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import distance


def pre_process(x: str) -> str:
    """ Pre process question strings for abbreviations and numbers
        Args:
            x: question string that is to be pre processed
        Returns:
            Returns a pre processed string
    """
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)

    return x


def get_token_features(q1: str, q2: str, safe_div: float, stop_words: Set[str]) -> List[float]:
    """ Computes a list of token features for two question strings
            Args:
                q1: String for question 1
                q2: String for question 2
                safe_div: A small float to avoid division by zero
                stop_words: set of nltk stop words
            Returns:
                A list of token features for two question strings
    """
    token_features = [0.0] * 10

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in stop_words])
    q2_words = set([word for word in q2_tokens if word not in stop_words])

    q1_stops = set([word for word in q1_tokens if word in stop_words])
    q2_stops = set([word for word in q2_tokens if word in stop_words])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + safe_div)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + safe_div)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + safe_div)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + safe_div)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + safe_div)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + safe_div)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens)) / 2

    return token_features


def get_longest_substr_ratio(a: str, b: str) -> float:
    """ Computes the ration of longest common substring length and the length of the smaller string
        Args:
            a: String for question 1
            b: String for question 2
        Returns:
            Returns longest common substring ration for two question strings
    """
    strings = list(distance.lcsubstrings(a, b))
    if len(strings) == 0:
        return 0
    else:
        return len(strings[0]) / (min(len(a), len(b)) + 1)


def extract_features(df: pd.DataFrame, safe_div: float, stop_words: Set[str]) -> pd.DataFrame:
    """ Helper function to extract nlp features from one given data sets
            Args:
                df: Train or test DataFrame from which nlp features are to be extracted
                safe_div: small float value to bypass division by zero
                stop_words: Set of english stop words from nltk

            Returns:
                Input DataFrame with columns added fro nlp features
    """
    df["question1"] = df["question1"].fillna("").apply(pre_process)
    df["question2"] = df["question2"].fillna("").apply(pre_process)

    print("token features...")
    token_features = df.apply(lambda x: get_token_features(
        x["question1"], x["question2"], safe_div, stop_words), axis=1)
    df["cwc_min"] = list(map(lambda x: x[0], token_features))
    df["cwc_max"] = list(map(lambda x: x[1], token_features))
    df["csc_min"] = list(map(lambda x: x[2], token_features))
    df["csc_max"] = list(map(lambda x: x[3], token_features))
    df["ctc_min"] = list(map(lambda x: x[4], token_features))
    df["ctc_max"] = list(map(lambda x: x[5], token_features))
    df["last_word_eq"] = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"] = list(map(lambda x: x[8], token_features))
    df["mean_len"] = list(map(lambda x: x[9], token_features))

    print("fuzzy features..")
    df["token_set_ratio"] = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"] = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"] = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"] = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)

    return df


def nlp_feature_extractor(df_train: pd.DataFrame,
                          df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Main function that extract nlp features from train and test data sets
        Args:
            df_train: DataFrame for Training data
            df_test: DataFrame for Test data
        Returns:
            Tuple of training and test DataFrames with nlp feature columns
    """
    safe_div = 0.0001
    stop_words = stopwords.words("english")

    print("Extracting features for train:")
    train_df = extract_features(df_train, safe_div, stop_words)
    train_df.drop(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"], axis=1, inplace=True)

    print("Extracting features for test:")
    test_df = extract_features(df_test, safe_div, stop_words)
    test_df.drop(["test_id", "question1", "question2"], axis=1, inplace=True)

    return train_df, test_df
