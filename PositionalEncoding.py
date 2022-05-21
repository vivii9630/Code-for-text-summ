import math
import torch
import numpy as np
from torch import nn, Tensor
import matplotlib.pyplot as plt


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)



class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pe_final = pos_embedding
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.max_l = maxlen

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    def out(self):
        return self.pe_final

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


    def plotSinusoid(self, k, n=10000):
        x = np.arange(0, 100, 1)
        denominator = np.power(n, 2 * x / self.max_l)
        y = np.sin(k / denominator)
        plt.plot(x, y)
        plt.title('k = ' + str(k))
        plt.show()

    def visualizePE(self):
        plt.figure(figsize=(20, 8))
        plt.pcolormesh(self.out())
        plt.xlabel('Position Embeddings')
        plt.ylabel('Token Position')
        plt.colorbar()
        plt.show()

#Testing
if __name__ == '__main__':
    inp_seq = np.arange(0, 500,1)

    pos = PositionalEncoding(10012, 0.3, 150)

    fig = plt.figure(figsize=(15, 4))
    for i in range(4):
        plt.subplot(141 + i)
        pos.plotSinusoid(i * 4)

    """cax = plt.matshow(pe_matrix)
    plt.gcf().colorbar(cax)
    plt.figure(figsize=(15, 4))
    plt.show()"""

    pos.visualizePE()

    #P = getPositionEncoding(seq_len=100, d=512, n=10000)
