""" Code file to extract non-nlp features from the questions """

from typing import Dict, Mapping, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx


def create_question_hash(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Mapping[int, str]:
    """ Creates a hash set of all non duplicate questions from train and test set
        Args:
            train_df: DataFrame for Training data
            test_df: DataFrame for Test data
        Returns:
            Dictionary of all questions from train and test data
    """
    train_qs = np.dstack([train_df["question1"], train_df["question2"]]).flatten()
    test_qs = np.dstack([test_df["question1"], test_df["question2"]]).flatten()
    all_qs = np.append(train_qs, test_qs)
    all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
    all_qs.reset_index(inplace=True, drop=True)
    question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()

    return question_dict


def get_hash(df: pd.DataFrame, hash_dict: Mapping[int, str]) -> pd.DataFrame:
    """ Replaces questions in the DataFrame by their hash indexes
            Args:
                df: Training or Test DataFrame
                hash_dict: Hash Set for all questions in training and test data
            Returns:
                Initial DataFrame where questions are replaced by their hash index
    """
    df["qid1"] = df["question1"].map(hash_dict)
    df["qid2"] = df["question2"].map(hash_dict)

    return df.drop(["question1", "question2"], axis=1)


def get_kcore_dict(df: pd.DataFrame, nb_cores: int) -> pd.DataFrame:
    """ Computes the degree of question nodes for a given question column up to a maximum node degree
        Args:
            df: DataFrame containing all questions from train and test data sets
            nb_cores: Maximal node degree to be considered
        Returns:
            Returns a DataFrame with one column for node degrees
    """
    g = nx.Graph()
    g.add_nodes_from(df.qid1)
    edges = list(df[["qid1", "qid2"]].to_records(index=False))
    g.add_edges_from(edges)
    g.remove_edges_from(g.selfloop_edges())

    df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
    df_output["kcore"] = 0
    for k in range(2, nb_cores + 1):
        ck = nx.k_core(g, k=k).nodes()
        print("kcore", k)
        df_output.ix[df_output.qid.isin(ck), "kcore"] = k

    return df_output.to_dict()["kcore"]


def get_kcore_features(df: pd.DataFrame, kcore_dict: pd.DataFrame) -> pd.DataFrame:
    """ Computes the degree of question nodes for for both question columns up to a maximum node degree
         Args:
             df: Train or test DataFrame
             kcore_dict: DataFrame of node degrees
         Returns:
             Returns a DataFrame with one column each for node degrees of question1 and question2
     """
    df["kcore1"] = df["qid1"].apply(lambda x: kcore_dict[x])
    df["kcore2"] = df["qid2"].apply(lambda x: kcore_dict[x])

    return df


def convert_to_minmax(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """ Computes the minimum and maximum values for each question pair row for a given non-nlp feature
             Args:
                 df: Train or test DataFrame
                 col: Feature column name for which the min and max value are being calculated
             Returns:
                 Returns a DataFrame with one column each for minimum and maximum values of the feature
         """
    sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
    df["min_" + col] = sorted_features[:, 0]
    df["max_" + col] = sorted_features[:, 1]

    return df.drop([col + "1", col + "2"], axis=1)


def get_neighbors(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[int, set]:
    """ Computes a dictionary of neighbour set of all questions that a given question is linked to
        Args:
            train_df: DataFrame for Training data
            test_df: DataFrame for Test data
        Returns:
            Dictionary of neighbours for each question
    """
    neighbors = defaultdict(set)
    for df in [train_df, test_df]:
        for q1, q2 in zip(df["qid1"], df["qid2"]):
            neighbors[q1].add(q2)
            neighbors[q2].add(q1)

    return neighbors


def get_neighbor_features(df: pd.DataFrame, neighbors: Dict[int, set],
                          neighbour_upper_bound: int) -> pd.DataFrame:
    """ Computes common neighbor ratio and common neighbor count for each question pair
        Args:
            df: Train or test DataFrame
            neighbors: Dictionary of neighbours for each question
            neighbour_upper_bound: Upper bound defined for number of neighbor to check outliers
        Returns:
            Initial Dictionary with added columns for neighbor features
    """
    common_nc = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
    min_nc = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
    df["common_neighbor_ratio"] = common_nc / min_nc
    df["common_neighbor_count"] = common_nc.apply(lambda x: min(x, neighbour_upper_bound))

    return df


def get_freq_features(df: pd.DataFrame, frequency_map: Dict[str, int],
                      freq_upper_bound: int) -> pd.DataFrame:
    """ Computes minimum and maximum frequency for each question pair
        Args:
            df: Train or test DataFrame
            frequency_map: Dictionary of frequency for each question in train and test DataFrames
            freq_upper_bound: Upper bound defined for teh frequency of questions
        Returns:
            Initial Dictionary with added columns for min and max frequency features
    """
    df["freq1"] = df["qid1"].map(lambda x: min(frequency_map[x], freq_upper_bound))
    df["freq2"] = df["qid2"].map(lambda x: min(frequency_map[x], freq_upper_bound))

    return df


def non_nlp_feature_extractor(train_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Main function that extract nlp features from train and test data sets
        Args:
            train_df: DataFrame for Training data
            test_df: DataFrame for Test data
        Returns:
            Tuple of training and test DataFrames with nlp feature columns
    """

    nb_cores = 10
    freq_upper_bound = 100
    neighbour_upper_bound = 5

    print("Hashing the questions...")
    question_dict = create_question_hash(train_df, test_df)
    train_df = get_hash(train_df, question_dict)
    test_df = get_hash(test_df, question_dict)
    print("Number of unique questions:", len(question_dict))

    print("Calculating kcore features...")
    all_df = pd.concat([train_df, test_df])
    kcore_dict = get_kcore_dict(all_df, nb_cores)
    train_df = get_kcore_features(train_df, kcore_dict)
    test_df = get_kcore_features(test_df, kcore_dict)
    train_df = convert_to_minmax(train_df, "kcore")
    test_df = convert_to_minmax(test_df, "kcore")

    print("Calculating common neighbor features...")
    neighbors = get_neighbors(train_df, test_df)
    train_df = get_neighbor_features(train_df, neighbors, neighbour_upper_bound)
    test_df = get_neighbor_features(test_df, neighbors, neighbour_upper_bound)

    print("Calculating frequency features...")
    frequency_map = dict(zip(*np.unique(np.vstack((all_df["qid1"], all_df["qid2"])), return_counts=True)))
    train_df = get_freq_features(train_df, frequency_map, freq_upper_bound)
    test_df = get_freq_features(test_df, frequency_map, freq_upper_bound)
    train_df = convert_to_minmax(train_df, "freq")
    test_df = convert_to_minmax(test_df, "freq")

    cols = ["min_kcore", "max_kcore", "common_neighbor_count", "common_neighbor_ratio", "min_freq", "max_freq"]

    train_df = train_df.loc[:, cols]
    test_df = test_df.loc[:, cols]

    return train_df, test_df
