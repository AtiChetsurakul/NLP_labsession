from torchtext.data.utils import get_tokenizer
import torch
import torch.nn as nn
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import pickle
from predickapp.predickModule import LSTMLanguageModel
from predickapp.dickUtil import generate
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
nlp = spacy.load('en_core_web_md')
with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(vocab)
emb_dim = 400                # 400 in the paper
hid_dim = 1150               # 1150 in the paper
num_layers = 3                # 3 in the paper
dropout_rate = 0.5
lr = 1e-3

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim,
                          num_layers, dropout_rate).to(device)

model.load_state_dict(torch.load('predictor_weight.pt'))

prompt = 'for i in'
max_seq_len = 30
seed = 3407
# temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
# sample from this distribution higher probability will get more change
# for temperature in temperatures:
generation = generate(prompt, max_seq_len, .8, model, tokenizer,
                      vocab, device, seed)
print(str(.8)+'     '+' '.join(generation))

dickle = (generate, LSTMLanguageModel(vocab_size, emb_dim, hid_dim,
                                      num_layers, dropout_rate), tokenizer, nlp, device, vocab_size, vocab, seed)
with open('dicpackage.atikeep', 'wb') as handle:
    pickle.dump(dickle, handle)
print('dump success')
