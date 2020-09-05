""" Main Code file to get predictions to submit on Kaggle"""

import pandas as pd
import nltk
from pre_process_nlp_features import nlp_feature_extractor
from pre_process_non_nlp_features import non_nlp_feature_extractor
from train import run_lstm_with_embeddings
from post_process import post_process_results

# Train and test data frames
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

# Get NLP features
train_nlp_feats_df, test_nlp_feats_df = nlp_feature_extractor(df_train, df_test)
train_nlp_feats_df.to_csv("data/nlp_features_train.csv", index=False)
test_nlp_feats_df.to_csv("data/nlp_features_test.csv", index=False)

# Get Non-NLP features
train_non_nlp_feats_df, test_non_nlp_feats_df = non_nlp_feature_extractor(df_train, df_test)
train_non_nlp_feats_df.to_csv("data/non_nlp_features_train.csv", index=False)
test_non_nlp_feats_df.to_csv("data/non_nlp_features_test.csv", index=False)

# Run model
nltk.download('wordnet')
run_lstm_with_embeddings(df_train, df_test)

# Post process model predictions and create submission file
submission = post_process_results(df_train, df_test)
submission.to_csv("predictions/submission.csv", index=False)
