import torch
import torch.nn as nn


class GloVe(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(
            vocab_size, embed_size)  # center embedding
        self.embedding_u = nn.Embedding(
            vocab_size, embed_size)  # out embedding

        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, coocs, weighting):
        center_embeds = self.embedding_v(
            center_words)  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words)  # [batch_size, 1, emb_size]

        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(
            center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        # note that coocs already got log
        loss = weighting*torch.pow(inner_product +
                                   center_bias + target_bias - coocs, 2)

        return torch.sum(loss)


class SkipgramNegSampling(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(
            vocab_size, emb_size)  # center embedding
        self.embedding_u = nn.Embedding(vocab_size, emb_size)  # out embedding
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(
            center_words)  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words)  # [batch_size, 1, emb_size]
        # [batch_size, num_neg, emb_size]
        neg_embeds = -self.embedding_u(negative_words)

        positive_score = target_embeds.bmm(
            center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        negative_score = neg_embeds.bmm(center_embeds.transpose(1, 2))
        # [batch_size, k, emb_size] @ [batch_size, emb_size, 1] = [batch_size, k, 1]

        loss = self.logsigmoid(positive_score) + \
            torch.sum(self.logsigmoid(negative_score), 1)

        return -torch.mean(loss)

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds


class Skipgram(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.embedding_u = nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, target_words, all_vocabs):
        center_embeds = self.embedding_v(
            center_words)  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(
            target_words)  # [batch_size, 1, emb_size]
        # [batch_size, voc_size, emb_size]
        all_embeds = self.embedding_u(all_vocabs)

        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, voc_size, emb_size] @ [batch_size, emb_size, 1] = [batch_size, voc_size, 1] = [batch_size, voc_size]

        nll = -torch.mean(torch.log(torch.exp(scores) /
                          torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)))  # log-softmax
        # scalar (loss must be scalar)

        return nll  # negative log likelihood


class CBOW(nn.Module):  # same as skipgram
    def __init__(self, voc_size, emb_size):
        super(CBOW, self).__init__()
        # is a lookup table mapping all ids in voc_size, into some vector of size emb_size
        self.embedding_center_word = nn.Embedding(voc_size, emb_size)
        self.embedding_outside_word = nn.Embedding(voc_size, emb_size)

    def forward(self, center_word, outside_word, all_vocabs):
        # center_word, outside_word: (batch_size,1)
        #all_vocabs : (batch_size, voc_size)
        # convert them into embedding
        center_word_embed = self.embedding_center_word(
            center_word)  # v_c (batch_size,1, emb_size)
        outside_word_embed = self.embedding_outside_word(
            outside_word)  # u_o (batch_size,1, emb_size)
        all_vocabs_embed = self.embedding_outside_word(
            all_vocabs)  # u_w (batch_size,voc_size, emb_size)
        # print(center_word_embed.shape,outside_word_embed.shape,all_vocabs_embed.shape)
        # bmm is basically @ or .dot but across batches (ie., ignore the batch dimension)
        top_term = outside_word_embed.bmm(
            center_word_embed.transpose(1, 2)).squeeze(2)
        # (batch_size,1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) ===> (batch_size, 1)
        top_term_exp = torch.exp(top_term)  # exp(uo vc)
        #(batch_size, 1)
        lower_term = all_vocabs_embed.bmm(
            center_word_embed.transpose(1, 2)).squeeze(2)
        # (batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) ===> (batch_size, voc_size)
        lower_term_sum = torch.sum(torch.exp(lower_term))  # sum exp(uw, vc)
        #(batch_size, 1)
        loss_fn = -torch.mean(torch.log(top_term_exp/lower_term_sum))
        # (batc_size,1) / (batch_size,1) ==mena==> scalar
        return loss_fn
