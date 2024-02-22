import torch

CPU = torch.device('cpu')
GPU = torch.device('cuda:0')
USE_GPU = True

class Model(torch.nn.Module):
    def __init__(self, vocab_sz, embedding_dim, hidden_sz):
        super(Model, self).__init__()
        self.H = hidden_sz
        self.num_layers = 1
        self.D = 1
        self.embed = torch.nn.Embedding(vocab_sz, embedding_dim)
        self.dropout_l = torch.nn.Dropout(p=0.2)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_sz)
        self.fc = torch.nn.Linear(hidden_sz, vocab_sz)  

    def forward(self, corpus, seq_lengths, max_len):
        #  N, L
        N = corpus.shape[0]
        embed_out = self.embed(corpus)  # N, L, emb_dim
        embed_out = embed_out.permute(1, 0, 2)  # L, N, emb_dim
        
        lstm_inputs = torch.nn.utils.rnn.pack_padded_sequence(embed_out, lengths=seq_lengths)
        
        h0 = torch.zeros(self.D * self.num_layers, N, self.H)
        c0 = torch.zeros_like(h0)
        if USE_GPU:
            h0, c0 = h0.to(GPU), c0.to(GPU)
        lstm_out, (hn, cn) = self.lstm(lstm_inputs, (h0, c0))  # L, N, H
        lstm_out, lengths_unpacked = torch.nn.utils.rnn.pad_packed_sequence(sequence=lstm_out, total_length=max_len)
        
        lstm_out = self.dropout_l(lstm_out) # add l

        fc_out = self.fc(lstm_out)  # L, N, vocab_sz
        outs = fc_out.permute(1, 0, 2)  # N, L, vocab_sz

        return outs
    
    def predict(self, word2id, id2word, start_words, total_length):
        start_len = len(start_words)
        cur_idxs = [word2id[start_word] for start_word in start_words]

        h0 = torch.zeros(self.D * self.num_layers, 1, self.H)
        c0 = torch.zeros_like(h0)
        if USE_GPU:
            h0, c0 = h0.to(GPU), c0.to(GPU)
        
        for i in range(total_length-start_len):
            corpus = torch.LongTensor(cur_idxs)
            if USE_GPU:
                corpus = corpus.to(GPU)  
            corpus = corpus.reshape(1, -1)  # N, L

            embed_out = self.embed(corpus)  # N, L, emb_dim
            embed_out = embed_out.permute(1, 0, 2)  # L, N, emb_dim

            lstm_out, (hn, cn) = self.lstm(embed_out, (h0, c0))  # L, N, H
            h0, c0 = hn, cn

            fc_out = self.fc(lstm_out)  # L, 1, vocab_sz
            outs = fc_out.permute(1, 0, 2)  # 1, L, vocab_sz

            # 1, L, vocab_sz
            _, pred_idxs = outs.sort(dim=2, descending=True)
            pred_idx = pred_idxs[0][i][0] if pred_idxs[0][i][0] > 0 else pred_idxs[0][i][1]
            cur_idxs.append(pred_idx.item())
        
        words = []
        for idx in cur_idxs:
            words.append(id2word[idx])
        
        for i, word in enumerate(words, 0):
            print(word, end='')
            if i == 6 or i == 20:
                print('，', end='\n')
            elif i == 13 or i == 27:
                print('。', end='\n')
        

