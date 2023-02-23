import pickle
import torch
with open('dicpackage.atikeep', 'rb') as handle:

    generate, model, tokenizer, nlp, device, vocab_size, vocab, seed = pickle.load(
        handle)

model.to(device).load_state_dict(torch.load('predictor_weight.pt'))

prompt = 'for i in'
# temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
# sample from this distribution higher probability will get more change
# for temperature in temperatures:
generation = generate(prompt, 30, .8, model, tokenizer,
                      vocab, device, seed)
print(' '.join(generation))
