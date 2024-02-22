import torch
from torchtext.vocab import Vectors
import numpy as np

from utils import *
from model import *

BATCH_SIZE = 256
EPOCH = 30
TRAIN_FILE = './train.txt'
TEST_FILE = './test.txt'
EMBEDDING_DIM = 100
HIDDEN_SIZE = 100

CPU = torch.device('cpu')
GPU = torch.device('cuda:0')
USE_GPU = True

word2id, id2word, label2id = None, None, None

def train(epoch, myloader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for i, (sentences, labels) in enumerate(myloader, 0):
        corpus, labels, seq_lengths, max_length = make_tensor(word2id, id2word, label2id, sentences, labels)
        corpus, labels = corpus.to(GPU), labels.to(GPU)
        outs = (-1) * model(corpus, labels, seq_lengths, max_length)
        loss = outs.mean(dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch={epoch}, mean loss = {total_loss / len(myloader)}')
    return total_loss / len(myloader)

def test(epoch, myloader, model, len_data, num_labels):
    model.eval()
    with torch.no_grad():
        sum_val = 0
        for i, (sentences, labels) in enumerate(myloader, 0):
            corpus, labels, seq_lengths, max_length = make_tensor(word2id, id2word, label2id, sentences, labels)
            corpus, labels = corpus.to(GPU), labels.to(GPU)
            N = corpus.shape[0]
            outs = model.predict(corpus, seq_lengths, max_length)

            # outs:list, (N, L_no_padding)
            # labels:LongTensor, (N, L_padding); seq_lengths (N,)
            labels = labels.tolist()
            for i in range(len(labels)):
                labels[i] = labels[i][:seq_lengths[i]]
            
            val_f1 = compute_f1(outs, labels, num_labels)
            sum_val += val_f1
            
        print(f'f1 score = {sum_val / len(myloader)}')
        return sum_val / len(myloader)

if __name__ == '__main__':
    train_sentences, train_labels = data_preprocess(TRAIN_FILE)
    test_sentences, test_labels = data_preprocess(TEST_FILE)
    word2id, id2word = create_dict(train_sentences+test_sentences)
    
    vectors = Vectors(name='glove.6B.100d.txt', cache='./glove.6B/')
    words = [word for word in word2id.keys()]
    vectors = vectors.get_vecs_by_tokens(words, lower_case_backup=True)
    
    label2id = {'B-PER':0, 'I-PER':1, 'B-LOC':2, 'I-LOC':3, 'B-ORG':4, 'I-ORG':5, 'B-MISC':6, 'I-MISC':7, 'O':8}
    num_labels = len(label2id)
    
    mydataset = MyDataSet(train_sentences, train_labels)
    myloader = DataLoader(dataset=mydataset, batch_size=BATCH_SIZE, shuffle=True)
    
    testDataset = MyDataSet(test_sentences, test_labels)
    testLoader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE, shuffle=False)
    len_data = len(testDataset)
    
    model = Model(len(word2id), EMBEDDING_DIM, HIDDEN_SIZE, num_labels, vectors=vectors)
    if USE_GPU:
        model = model.to(GPU)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999))

    loss_l = []
    f1_l = []
    for epoch in range(EPOCH):
       loss_l.append(train(epoch, myloader, model, criterion, optimizer))
       f1_l.append(test(epoch, testLoader, model, len_data, num_labels))
    
    draw_pict(loss_l)
    draw_pict(f1_l)
    