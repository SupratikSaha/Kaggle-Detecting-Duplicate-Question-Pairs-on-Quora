""" Code file to post process model predictions """

from collections import defaultdict
import numpy as np
import pandas as pd


def post_process_results(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """ Function that post processes model predictions by considering class imbalance ratio and duplicates
        Args:
            df_train: DataFrame for Training data
            df_test: DataFrame for Test data
        Returns:
            Post process DataFrame od test data set predictions ready for submission
    """
    num_models = 10
    train_target_mean = 0.37
    test_target_mean = 0.16
    repeat = 2
    dup_threshold = 0.5
    not_dup_threshold = 0.1
    max_update = 0.2
    dup_upper_bound = 0.98
    not_dup_upper_bound = 0.01

    print("Average Ensembling")
    df = pd.read_csv("predictions/preds0.csv")
    for i in range(1, num_models):
        df["is_duplicate"] = df["is_duplicate"] + \
                             pd.read_csv("predictions/preds" + str(i) + ".csv")["is_duplicate"]
    df["is_duplicate"] /= num_models

    print("Adjusting predictions considering the different class imbalance ratio...")
    a = test_target_mean / train_target_mean
    b = (1 - test_target_mean) / (1 - train_target_mean)
    df["is_duplicate"] = df["is_duplicate"].apply(lambda x: a * x / (a * x + b * (1 - x)))

    test_label = np.array(df["is_duplicate"])

    print("Updating the predictions of the pairs with common duplicates..")
    for i in range(repeat):
        dup_neighbors = defaultdict(set)

        for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]):
            if dup:
                dup_neighbors[q1].add(q2)
                dup_neighbors[q2].add(q1)

        for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]):
            if dup > dup_threshold:
                dup_neighbors[q1].add(q2)
                dup_neighbors[q2].add(q1)

        count = 0
        for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])):
            dup_neighbor_count = len(dup_neighbors[q1].intersection(dup_neighbors[q2]))
            if dup_neighbor_count > 0 and test_label[index] < dup_upper_bound:
                update = min(max_update, (dup_upper_bound - test_label[index]) / 2)
                test_label[index] += update
                count += 1

        print("Updated:", count)

    print("Updating the predictions of the pairs with common non-duplicates..")
    for i in range(repeat):
        not_dup_neighbors = defaultdict(set)

        for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]):
            if not dup:
                not_dup_neighbors[q1].add(q2)
                not_dup_neighbors[q2].add(q1)

        for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]):
            if dup < not_dup_threshold:
                not_dup_neighbors[q1].add(q2)
                not_dup_neighbors[q2].add(q1)

        count = 0
        for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])):
            dup_neighbor_count = len(not_dup_neighbors[q1].intersection(not_dup_neighbors[q2]))
            if dup_neighbor_count > 0 and test_label[index] > not_dup_upper_bound:
                update = min(max_update, (test_label[index] - not_dup_upper_bound) / 2)
                test_label[index] -= update
                count += 1

        print("Updated:", count)

    submission = pd.DataFrame({"test_id": df_test["test_id"], "is_duplicate": test_label})

    # Make sure we have the correct ids
    submission_final = submission[submission['test_id'].apply(lambda x: isinstance(x, (int, np.int64)))]

    # Average out values if there are ids with multiple values
    submission_final = submission_final.groupby('test_id', as_index=False).mean()

    return submission_final
