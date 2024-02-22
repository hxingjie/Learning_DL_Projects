import torch
import numpy as np

from TorchCRF import CRF

CPU = torch.device('cpu')
GPU = torch.device('cuda:0')
USE_GPU = True

class Model(torch.nn.Module):
    def __init__(self, vocab_sz, embedding_dim, hidden_sz, num_labels, vectors=None):
        super(Model, self).__init__()
        self.H = hidden_sz // 2
        if vectors == None:
            self.embed = torch.nn.Embedding(vocab_sz, embedding_dim=embedding_dim)
        else:
            self.embed = torch.nn.Embedding.from_pretrained(vectors)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_sz // 2, bidirectional=True)
        self.dropout_l = torch.nn.Dropout(p=0.2) # add l
        self.fc = torch.nn.Linear(in_features=hidden_sz, out_features=num_labels)
        self.crf = CRF(num_labels)

    def get_mask(self, seq_lengths, max_length):
        mask = []
        for seq_length in seq_lengths:
            # 3
            mask.append([True for _ in range(seq_length)] + [False for _ in range(max_length - seq_length)])
        mask = torch.BoolTensor(mask)
        if USE_GPU:
            mask = mask.to(GPU)
        return mask

    def lstm_layer(self, sentences, seq_lengths):
        #  N, L
        N = sentences.shape[0]
        embed_out = self.embed(sentences)  # N, L, emb_dim
        embed_out = embed_out.permute(1, 0, 2)  # L, N, emb_dim
        
        lstm_inputs = torch.nn.utils.rnn.pack_padded_sequence(embed_out, lengths=seq_lengths)
        
        h0 = torch.zeros(2, N, self.H)
        c0 = torch.zeros_like(h0)
        if USE_GPU:
            h0, c0 = h0.to(GPU), c0.to(GPU)
        lstm_out, (hn, cn) = self.lstm(lstm_inputs, (h0, c0))  # L, N, H
        lstm_out, lengths_unpacked = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        
        lstm_out = self.dropout_l(lstm_out) # add l

        fc_out = self.fc(lstm_out)  # L, N, num_labels
        fc_out = fc_out.permute(1, 0, 2)  # N, L, num_labels

        return fc_out

    def forward(self, sentences, labels, seq_lengths, max_length):
        fc_out = self.lstm_layer(sentences, seq_lengths)  # L, N, num_labels
        out = self.crf(fc_out, labels, self.get_mask(seq_lengths, max_length))

        return out
    
    def predict(self, sentences, seq_lengths, max_length):
        fc_out = self.lstm_layer(sentences, seq_lengths)
        return self.crf.viterbi_decode(fc_out, self.get_mask(seq_lengths, max_length))
             
