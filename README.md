# Kaggle Quora Question Pairs Competition Solution

## Kaggle Competition link

https://www.kaggle.com/c/quora-question-pairs

## Submission Details

The submission had a log loss score of 0.13499 on the private leader board 
and 0.13264 on the public leader board

![My Submission](images/kaggle_submission_score.jpg)

# Steps to run project code

## Packages to be installed

Following packages specified in requirements.txt file need to be installed
Distance==0.1.3
fuzzywuzzy==0.17.0
keras==2.2.4
networkx==1.11
nltk==3.4
numpy==1.16.2
pandas==0.24.2
scikit-learn==0.22.2.post1

## Folders needed to run code

Create folders named data and predictions in the project directory

## Download Competition Data

Download the competition data files from [Kaggle Competition Data Link](https://www.kaggle.com/c/quora-question-pairs/data) and place them in the data folder

## Download Glove Embedding Vector

Download the glove vector into project directory [Pre-Trained Glove Word Vector](https://nlp.stanford.edu/projects/glove)
 
## Running Code

Run the \_\_main__.py file. It is advised to run the code in pieces. For reference, it took me about 2 days on my HP Spectre i5

# Model

- Model uses in total 25 nlp and non-nlp features
- A 10-fold validation strategy was used
- A Glove embedding vector and a LSTM model was used to get predictions
- Rare words in questions were replaced by an invalid word indicator "memento"
- Average ensembling is used to derive final model predictions