import pandas as pd
import numpy as np

def load_data():
    np.random.seed(1)
    df = pd.read_csv('./data/train.csv', usecols=["question1","question2","is_duplicate"])

    df = df.sample(frac=1).reset_index(drop=True)
    train_index = np.random.randint(0, 10, df.shape[0]) > 2

    train = df[train_index]
    test = df[train_index == False]
    print(train.shape, test.shape)

    train_X, train_Y = train[['question1', 'question2']].values, train['is_duplicate'].values
    test_X, test_Y = test[['question1', 'question2']].values, test['is_duplicate'].values
    return train_X, train_Y, test_X, test_Y
