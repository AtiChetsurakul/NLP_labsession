# numpy version
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import torch
import numpy as np


def cos_sim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def get_embed_test(word, word2index, model):
    try:
        index = word2index[word]
    except:
        index = word2index['<UNK>']  # unknown
    word = torch.LongTensor([index])

    embed = (model.embedding_v(word) + model.embedding_u(word))/2
    return np.array(embed[0].detach().numpy())


def get_embed_test_c_Bro(word, word2index, model):
    try:
        index = word2index[word]
    except:
        index = word2index['<UNK>']  # unknown
    word = torch.LongTensor([index])

    embed = (model.embedding_center_word(word) +
             model.embedding_outside_word(word))/2
    return np.array(embed[0].detach().numpy())


def find_analogy(a, b, c, vocab, get_embed_test, word2index, model):
    emb_a, emb_b, emb_c = get_embed_test(
        a, word2index, model), get_embed_test(b, word2index, model), get_embed_test(c, word2index, model)
    vector = emb_b - emb_a + emb_c
    similarity = -1
    retvoc = ''
    for voc in vocab:
        if voc not in [a, b, c]:
            current_sim = cos_sim(
                vector, get_embed_test(voc, word2index, model))
            if current_sim > similarity:
                similarity = current_sim  # update better one
                ret_voc = voc
                # print('hi')
    return ret_voc, similarity


def analogy_accuracy(data, vocab, get_embed_test, word2index, model):  # testing
    corrects = 0
    total = len(data)
    for a, b, c, d in data:
        # a, b, c, d = row['A'],row['B'],row['C'],row['D']
        predict = find_analogy(
            a, b, c, vocab, get_embed_test, word2index, model)[0]
        if predict == d:
            corrects += 1
    acc = corrects/total
    return acc


def analogy_accuracy_(data, model):  # testing
    corrects = 0
    total = len(data)
    check = [0, 0, 0, 0, 0]
    for ind, (a, b, c, d) in enumerate(data):

        # a, b, c, d = row['A'],row['B'],row['C'],row['D']
        predict = model.most_similar(positive=[c, b], negative=[a])[0][0]
        if predict == d:
            corrects += 1
            if ind < 20:
                check[0] += 1
            elif ind < 40:
                check[1] += 1
            elif ind < 60:
                check[2] += 1
            elif ind < 80:
                check[3] += 1
            else:
                check[-1] += 1
    # acc = corrects/total
    print(f'Count by categories {check}')
    return acc, check
