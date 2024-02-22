import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time


def text_preprocess(sentences):
    word2id = {'_':0}
    id2word = {0:'_'}
    corpus = []
    for sentence in sentences:
        sentence = sentence.lower()
        words = sentence.split(' ')
        cs = []
        for word in words:
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
            cs.append(word2id[word])
        corpus.append(cs)

    return word2id, id2word, corpus


data_train = pd.read_csv('./train.tsv', sep='\t')
data_test = pd.read_csv('./test.tsv', sep='\t')

data_sentences = data_train['Phrase'].to_list()
data_sentiments = data_train['Sentiment'].to_list()

train_sentences = data_sentences[:100000]
train_sentiments = data_sentiments[:100000]

valid_sentences = data_sentences[100000:]
valid_sentiments = data_sentiments[100000:]

test_sentenceIds = data_test['SentenceId'].to_list()
test_sentences = data_test['Phrase'].to_list()

word2id, id2word, corpus = text_preprocess(data_sentences + test_sentences)  # 将训练集和测试集的单词构建成id形式    


class MyDataset(Dataset):
    def __init__(self, is_train):
        if is_train:
            self.sentences = train_sentences
            self.sentiments = train_sentiments 
        else:
            self.sentences = valid_sentences
            self.sentiments = valid_sentiments
        
        self.len = len(self.sentences)

    def __getitem__(self, idx) -> tuple:
        return self.sentences[idx], self.sentiments[idx]

    def __len__(self) -> int:
        return self.len
    

class TestDataset(Dataset):
    def __init__(self):
        self.test_sentences = test_sentences

        self.len = len(self.test_sentences)

    def __getitem__(self, idx):
        return self.test_sentences[idx]

    def __len__(self) -> int:
        return self.len


# Parameters
N_SENTIMENT = 5
HIDDEN_SIZE = 64
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCH = 200
    
USE_GPU = True
if USE_GPU:
    device = torch.device('cuda:0')

train_set = MyDataset(is_train=True)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_set = MyDataset(is_train=False)
valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_set = TestDataset()
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

vocab_size = len(word2id)  # 词汇量


def trans_device(tensor):
    if USE_GPU:
        tensor = tensor.to(device)
    return tensor


def sentence2corpus(sentences):
    corpus = []
    for sentence in sentences:
        sentence = sentence.lower()
        words = sentence.split(' ')
        cs = []
        for word in words:
            cs.append(word2id[word])
        corpus.append(cs)
    return corpus


def make_tensors(sentences, sentiments):
    corpus = sentence2corpus(sentences)
    lengths = torch.LongTensor([len(sentence) for sentence in corpus])
    sentiments = torch.LongTensor(sentiments)  # 将tensor的数据类型转换为64位整数类型（torch.int64）

    sequences_pad = torch.zeros(len(corpus), lengths.max()).long()
    for idx, (sequence, length) in enumerate(zip(corpus, lengths), 0):
        sequences_pad[idx][:length] = torch.LongTensor(sequence)

    lengths, sorted_idx = lengths.sort(dim=0, descending=True)
    sequences_pad = sequences_pad[sorted_idx]
    sentiments = sentiments[sorted_idx]

    return trans_device(sequences_pad), lengths, trans_device(sentiments)


def make_test_tensors(sentences):
    corpus = sentence2corpus(sentences)
    sequences: list = [sentence for sentence in corpus]
    lengths = torch.LongTensor([len(sentence) for sentence in corpus])

    sequences_pad = torch.zeros(len(sequences), lengths.max()).long()
    for idx, (sequence, length) in enumerate(zip(sequences, lengths), 0):
        sequences_pad[idx][:length] = torch.LongTensor(sequence)

    lengths, sorted_idx = lengths.sort(dim=0, descending=True)
    sequences_pad = sequences_pad[sorted_idx]

    return trans_device(sequences_pad), lengths


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=1, bidirectional=True, word_vecs=None):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.emb = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.emb.weight.data = word_vecs
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                                num_layers=n_layers, bidirectional=bidirectional)
        self.fc0 = torch.nn.Linear(in_features=hidden_size * self.n_directions,
                                   out_features=output_size)
    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return trans_device(hidden)

    def forward(self, inputs, seq_lengths):
        inputs = inputs.T  # inputs: (B, Seq) -> (Seq, B)

        batch_size = inputs.shape[1]
        hidden = self._init_hidden(batch_size)

        emb = self.emb(inputs)  # emb: (Seq, B, hidden_size)
        # pack_padded_sequence's parameter lengths must be on cpu if it is tensor
        gru_inputs = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths=seq_lengths)

        outputs, hidden = self.gru(gru_inputs, hidden)
        fc_input = hidden[-1] \
            if self.n_directions == 1 else torch.concat([hidden[-1], hidden[-2]], dim=1)

        fc_output = self.fc0(fc_input)
        return fc_output


def trainModel():
    correct = 0
    total_loss = 0
    for idx, (sentences, sentiments) in enumerate(train_loader, 1):
        sequences, lengths, targets = make_tensors(sentences, sentiments)
        outputs = classifier(sequences, lengths)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            pred = outputs.max(dim=1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()
    print(f'{correct}/{len(train_set)}')      
    print(f'Loss = {total_loss / len(train_loader)}')


def validModel():
    correct = 0
    total = len(test_set)
    with torch.no_grad():
        for idx, (sentences, sentiments) in enumerate(valid_loader, 1):
            sequences, lengths, targets = make_tensors(sentences, sentiments)
            outputs = classifier(sequences, lengths)
            pred = outputs.max(dim=1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Valid: Accuracy {correct}/{total} {percent}%')
        return correct


def testModel():
    with torch.no_grad():
        result = {'PhraseId': [], 'Sentiment': []}
        phreaseId = 156061
        for idx, sentences in enumerate(test_loader, 1):
            sequences, lengths = make_test_tensors(sentences)
            outputs = classifier(sequences, lengths)
            pred = outputs.max(dim=1, keepdim=True)[1]
            for sentiment in pred:
                result['PhraseId'].append(phreaseId)
                phreaseId += 1
                result['Sentiment'].append(sentiment.item())
            
    df = pd.DataFrame(result)
    df.to_csv('output.csv', index=False)


if __name__ == '__main__':
    word_vecs = torch.load('in_embed.pth')
    classifier = RNNClassifier(input_size=vocab_size, hidden_size=HIDDEN_SIZE, 
                               output_size=N_SENTIMENT, n_layers=N_LAYER, word_vecs=word_vecs)
    if USE_GPU:
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0002)

    start = time.time()
    print("Training for %d epochs " % N_EPOCH)
    for epoch in range(1, N_EPOCH+1):
        trainModel()
        #validModel()

    #testModel()