import numpy as np
from scipy import stats
import pandas as pd

# example
#a = get_embed_test('walk',word2index,neg_skp_model)
# b = get_embed_test('walk',word2index,neg_skp_model)
# print(a,b)
# print(call_score(a,b)/10)


def call_score(emb1, emb2):
    corr, _ = stats.spearmanr(emb1, emb2)
    return corr*10


def semi_get_data(path, islist=True):
    with open(path, mode='r') as o:
        frtdf = pd.DataFrame(o.readlines())
    secdf = []
    for i in (frtdf[0]):
        secdf.append(' '.join(str(x) for x in i.split()))
    semifinal = [sent.split(" ") for sent in secdf]
    if islist:
        return semifinal
    else:
        return pd.DataFrame(semifinal, columns=['x1', 'x2', 'y'])
