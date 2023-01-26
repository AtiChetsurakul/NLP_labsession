
from nltk.corpus import brown

corpus_tokenize_ati_made_cannotdel_canu_haha = brown.sents(
    categories=['hobbies'])[:1500]
# print(corpus_tokenize_ati_made_cannotdel_canu_haha[-10])
corpus = [[j.lower() for j in i]
          for i in corpus_tokenize_ati_made_cannotdel_canu_haha]


def flatten(l):
    return [item for sublist in l for item in sublist]


vocab = list(set(flatten(corpus)))

word2index = {w: i for i, w in enumerate(vocab)}
voc_size = len(vocab)
print(voc_size)

vocab.append('<UNK>')

word2index['<UNK>'] = len(word2index)

index2word = {v: k for k, v in word2index.items()}

if False:
    with open('wordtotrain_use.atikeep', 'wb') as pic:
        pickle.dump((corpus, vocab, word2index, index2word), pic)
