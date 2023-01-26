# with open('questions-words.txt','r') as
import pandas as pd
import re
import numpy as np
import pickle
text = open('questions-words.txt', mode='r')
df_ = pd.DataFrame(text.readlines())
header = df_[0].str.startswith(':')
index_list = np.where(header)[0].tolist()
print(index_list)
# family
df_f = df_[8369:8389]
# gram2 Opposite
df_2 = df_[9869:9889]
# gram3 Comparative
df_3 = df_[10683:10703]
# gram4 Superlative
df_4 = df_[12016:12036]
# gram8 Plural
df_8 = df_[17356:17426]
df_c = pd.concat([df_f, df_2, df_3, df_4, df_8])


def clean_data(df_col):
    corpus = []
    for item in df_col:
        # remove special characters
        item = re.sub('[^A-Za-z0-9]+', ' ', str(item))
        item = item.lower()  # lower all characters
        item = item.split()  # split data
        corpus.append(' '.join(str(x) for x in item))
    return corpus


ctest = clean_data(df_c[0])
# data tokenized
tokenized_test = [sent.split(" ") for sent in ctest]
# tokenized_test[:5]
print(len(tokenized_test), 'success')
with open('data_to_test.atikeep', 'wb') as dic:
    pickle.dump(tokenized_test, dic)
