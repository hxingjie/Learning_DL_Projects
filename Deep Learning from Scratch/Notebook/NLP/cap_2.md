# CAP_2

## 1.基于计数的方法

### (1) 文本预处理

```python
import numpy as np

def text_preprocess(test):
    test = test.lower()  # 将文本全部变为小写字母
    test = test.replace('.', ' .')  # 在末尾的句号前添加空格
    words = test.split(' ')

    word2id = {}
    id2word = {}
    for word in words:
        if word not in word2id:
            idx = len(word2id)
            word2id[word] = idx
            id2word[idx] = word

    corpus = [word2id[word] for word in words]
    corpus = np.array(corpus)
    return corpus, word2id, id2word
```

### (2) 共现矩阵

```python
import numpy as np

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_idx in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                co_matrix[word_idx][corpus[left_idx]] += 1

            if right_idx < corpus_size:
                co_matrix[word_idx][corpus[right_idx]] += 1

    return co_matrix
```

### (3) 余弦相似度

```python
import numpy as np

def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    return np.dot(nx, ny)
```

### (4) 相似度搜索

```python
def most_similar(query, word2id, id2word, word_matrix, top=5):
    if query not in word2id.keys():
        print("%s is not found" % query)
        return

    query_id = word2id[query]
    query_vec = word_matrix[query_id]

    similarity = np.zeros(len(word2id))
    for i in range(len(word2id)):
        similarity[i] = cos_similarity(query_vec, word_matrix[i])

    cnt = 0
    for i in (-1*similarity).argsort():
        if i == query_id:
            continue
        print(f"{id2word[i]}: {similarity[i]}")

        cnt += 1
        if cnt >= top:
            return
```

## 2.基于计数的方法改进

### (1) ppmi

```python
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i]*S[j]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100+1) == 0:
                    print('%lf%% done' % (100*cnt/total))

    return M
```

```python
from matplotlib import pyplot as plt
from common.util import *

test = 'You say goodbye and I say hello.'
corpus, word2id, id2word = text_preprocess(test)

vocab_size = len(word2id)

co_matrix = create_co_matrix(corpus, vocab_size, 1)

most_similar('you', word2id, id2word, co_matrix)

W = ppmi(co_matrix)

U, S, V = np.linalg.svd(W)
print(co_matrix[0])
print(W[0])
print(U[0])
print(U[0][:2])

for word, word_id in word2id.items():
    plt.annotate(word, (U[word_id][0], U[word_id][1]))
    
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
```

### (2) PTB

```python
import numpy as np
from matplotlib import pyplot as plt
from common.util import *

from sklearn.utils.extmath import randomized_svd
import ptb

corpus, word2id, id2word = ptb.load_data('train')
window_size = 2
wordvec_size = 100
vocab_size = len(word2id)

C = create_co_matrix(corpus, vocab_size=vocab_size, window_size=2)
W = ppmi(C, verbose=True)

U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']

print()
for query in querys:
    print(f'{query}')
    most_similar(query, word2id, id2word, word_vecs, 5)
    print()
```

