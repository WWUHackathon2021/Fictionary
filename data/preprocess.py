import pandas as pd
from sklearn.model_selection import train_test_split
import re

df = pd.read_csv('dictionary.csv', header=None, names=['word', 'pos', 'definition'])
df = df[['word', 'definition']]

train, valid = train_test_split(df, train_size=0.9, random_state=1)

splits = {'train.csv': train, 'valid.csv': valid}

for k,v in splits.items():
    v.to_csv(k, sep='\t', index=False)
