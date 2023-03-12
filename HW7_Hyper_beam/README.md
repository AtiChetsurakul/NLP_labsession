# HOMEWORK 7

## model Transformer

- #### CODE Notebook

  - File:`HyperBeam.ipynb`
  - It hard to read from notebook, so I summarize it.

### CODE copilot

- #### model

``` py
from queue import PriorityQueue
from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, 
                 pf_dim, dropout, device,max_length = 100,PAD_IDX=PAD_IDX):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.pad_idx = PAD_IDX
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg,src, enc_src=[]):
        # print(trg.size(),src.size())
        # break
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        # trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        # trg_len = trg.shape[1]
        # trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]

        # trg_mask = trg_pad_mask & trg_sub_mask
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)&torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)          
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            # if src.size() == trg.size():
                # print('ahoy')
                # break
            # print(trg.size(),src.size())
            trg, attention = layer(trg, src, trg_mask, src_mask)

            # trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # break
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        #output = [batch size, trg len, output dim]
            
        return output, attention

    def decode(self, src, src_len, trg, method='beam-search'):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #src len = [batch size]

        # encoder_outputs, hidden = self.encoder(src, src_len) 
        #encoder_outputs = [src len, batch size, hid dim * 2]  (*2 because of bidirectional)(every hidden states)
        #hidden = [batch size, hid dim]  #final hidden state
        encoder_outputs = src
        hidden = hidden.unsqueeze(0)
        #hidden = [1, batch size, hid dim]
        
        if method == 'beam-search':
            return self.beam_decode(src, trg, hidden, encoder_outputs)
        else:
            return self.greedy_decode(trg, hidden, encoder_outputs)

    def greedy_decode(self, trg, decoder_hidden, encoder_outputs, ):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        seq_len, batch_size = trg.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
        decoder_input = Variable(trg.data[0, :]).cuda()  # sos
        print(decoder_input.shape)
        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1)

        return 

    def beam_decode(self, src_tensor, target_tensor, decoder_hiddens, encoder_outputs=None):
        #src_tensor      = [src len, batch size]
        #target_tensor   = [trg len, batch size]
        #decoder_hiddens = [1, batch size, hid dim]
        #encoder_outputs = [src len, batch size, hid dim * 2]
        
        target_tensor = target_tensor.permute(1, 0)
        #target_tensor = [batch size, trg len]
        
        #how many parallel searches
        beam_width = 3
        
        #how many sentence do you want to generate
        topk = 1  
        
        #final generated sentence
        decoded_batch = []
                
        #Another difference is that beam_search_decoding has 
        #to be done sentence by sentence, thus the batch size is indexed and reduced to only 1.  
        #To keep the dimension same, we unsqueeze 1 dimension for the batch size.
        for idx in range(target_tensor.size(0)):  # batch_size
            
            #decoder_hiddens = [1, batch size, dec hid dim]
            decoder_hidden = decoder_hiddens[:, idx, :]
            #decoder_hidden = [1, dec hid dim]
            
            #encoder_outputs = [src len, batch size, enc hid dim * 2]
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)
            #encoder_output = [src len, 1, enc hid dim * 2]
            
            mask = self.create_mask(src_tensor[:, idx].unsqueeze(1))
            # print("mask shape: ", mask.shape)
            
            #mask = [1, src len]

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([SOS_IDX]).to(device)

            # Number of sentence to generate
            endnodes = []  #hold the nodes of EOS, so we can backtrack
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()  #this is a min-heap

            # start the queue
            nodes.put((-node.eval(), node))  #we need to put - because PriorityQueue is a min-heap
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                # score is log p divides by the length scaled by some constants
                score, n = nodes.get()
                            
                # wordid is simply the numercalized integer of the word
                decoder_input  = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_IDX and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                # decoder_input = SOS_IDX
                # decoder_hidden = [1, hid dim]
                # encoder_output = [src len, 1, hid dim * 2]
                # mask = [1, src len]
                
                prediction, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output, mask)
                #prediction     = [1, output dim]  #1 because the batch size is 1
                #decoder hidden = [1, hid dim]

                #so basically prediction is probabilities across all possible vocab
                #we gonna retrieve k top probabilities (which is defined by beam_width) and their indexes
                #recall that beam_width defines how many parallel searches we want
                log_prob, indexes = torch.topk(prediction, beam_width)
                # log_prob      = (1, beam width)
                # indexes       = (1, beam width)
                
                nextnodes = []  #the next possible node you can move to

                # we only select beam_width amount of nextnodes
                for top in range(beam_width):
                    pred_t = indexes[0, top].reshape(-1)  #reshape because wordid is assume to be []; see when we define SOS
                    log_p  = log_prob[0, top].item()
                                    
                    #decoder hidden, previous node, current node, prob, length
                    node = BeamSearchNode(decoder_hidden, n, pred_t, n.logp + log_p, n.len + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # Once everything is finished, choose nbest paths, back trace them
            
            ## in case it does not finish, we simply get couple of nodes with highest probability
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            #look from the end and go back....
            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace by looking at the previous nodes.....
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]  #reverse it....
                utterances.append(utterance) #append to the list of sentences....

            decoded_batch.append(utterances)

        return decoded_batch  #(batch size, length)



class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        # self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, attention = self.self_attention(enc_src, enc_src, enc_src, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        # _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        # trg = self.enc_attn_layer_norm(trg)#+ self.dropout(_trg))            

        # trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))            
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention


```

### EVALUATE

- #### NORMAL

``` py
prompt = 'import '
max_seq_len = 5
seed = 0
        #superdiverse   more diverse
temperatures = [0.5, 0.7, 0.75, 0.8, 1.0] 
#sample from this distribution higher probability will get more change
for temperature in temperatures:
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
                          vocab, device, seed)
    print(str(temperature)+'\n'+' '.join(generation)+'\n')

'''
0.5
import import import import import import

0.7
import import layer layer layer layer

0.75
import import import layer layer layer

0.8
import import import layer layer layer

1.0
import import import layer layer layer

'''

```

- #### BEAMSEARH

``` py
model.decode('import',3,'import <><><><><>')

```

``` jupyter

UnboundLocalError                         Traceback (most recent call last)
Cell In[110], line 1
----> 1 model.decode('import',3,'import <><><><><>')

Cell In[92], line 79, in Decoder.decode(self, src, src_len, trg, method)
     69 def decode(self, src, src_len, trg, method='beam-search'):
     70     
     71     #src = [src len, batch size]
   (...)
     76     #encoder_outputs = [src len, batch size, hid dim * 2]  (*2 because of bidirectional)(every hidden states)
     77     #hidden = [batch size, hid dim]  #final hidden state
     78     encoder_outputs = src
---> 79     hidden = hidden.unsqueeze(0)
     80     #hidden = [1, batch size, hid dim]
     82     if method == 'beam-search':

UnboundLocalError: local variable 'hidden' referenced before assignment
```

- #### Greedy Search

``` py
model.decode('import',3,'import <><><><><>','กรี้ดดี้เสิด')

```

``` jupyter

UnboundLocalError                         Traceback (most recent call last)
Cell In[111], line 1
----> 1 model.decode('import',3,'import <><><><><>','กรี้ดดี้เสิด')

Cell In[92], line 79, in Decoder.decode(self, src, src_len, trg, method)
     69 def decode(self, src, src_len, trg, method='beam-search'):
     70     
     71     #src = [src len, batch size]
   (...)
     76     #encoder_outputs = [src len, batch size, hid dim * 2]  (*2 because of bidirectional)(every hidden states)
     77     #hidden = [batch size, hid dim]  #final hidden state
     78     encoder_outputs = src
---> 79     hidden = hidden.unsqueeze(0)
     80     #hidden = [1, batch size, hid dim]
     82     if method == 'beam-search':

UnboundLocalError: local variable 'hidden' referenced before assignment
```
