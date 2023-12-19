# Util

---

## 1.preprocess

```python
def preprocess(text: str) -> torch.Tensor:
    # 'you say hello and I say hi'
    text = text.lower()
    text = text.replace('.', ' .')
    text = text.split(' ')

    word2id = {}
    id2word = {}
    for word in text:
        if word not in word2id:
            idx = len(word2id)
            word2id[word] = idx
            id2word[idx] = word
    
    corpus = torch.Tensor([word2id[word] for word in text])
    return corpus, word2id, id2word
```

---

## 2.create_contexts_target

```python
def create_contexts_target(corpus, window_size):
    target = corpus[window_size, -window_size]

    contexts = []
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    
    return contexts, target
```

---

## 3.convert_one_hot

```python
def convert_one_hot(corpus, V):
    corpus = np.array(corpus, dtype=np.int32)
    N = corpus.shape[0]
    
    if corpus.ndim == 1:
        # [0, 1, 3, 4]
        new_corpus = np.zeros((N, V))
        for idx, word_id in enumerate(corpus):
            new_corpus[idx][word_id] = 1
        return new_corpus
    elif corpus.ndim == 2:
        # [[0, 1, 3, 4]]
        C = corpus.shape[1]
        new_corpus = np.zeros((N, C, V))
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                new_corpus[idx_0][idx_1][word_id] = 1
        return new_corpus
```

---

## 4.draw_pict

```python
def draw_pict(src_data:list=None, beg:int=0, end:int=None):
    x = np.arange(beg, end, 1, dtype=np.int32)  # [0, 6), 步长为0.1
    y = np.array(src_data)

    plt.plot(x, y, linestyle="-")
    plt.show()
```

---

## 5.get_neg_samples

```python
def get_neg_samples(corpus, target, id2word, neg_sz):
    vocab_size = len(id2word)
    word_freq = [0 for _ in range(vocab_size)]
    for idx in corpus:
        word_freq[idx] += 1
    
    word_freq = np.array(word_freq)
    word_freq = np.power(word_freq, 0.75)
    word_freq /= np.sum(word_freq)
    
    target = np.array(target)
    batch_size = target.shape[0]
    neg_samples = np.zeros((batch_size, neg_sz))
    for idx in range(batch_size):
        temp_p = word_freq.copy()
        temp_p[target[idx]] = 0
        temp_p /= np.sum(temp_p)
        neg_samples[idx, :] = np.random.choice(vocab_size, size=neg_sz, p=temp_p)
	return neg_samples
```

---

## 6. analogy

```python
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)  # 对应元素相乘后求和


def most_similar(query_word, word2id, id2word, word_vecs, top=5):
    # word_vecs: (V, H)  
    similar = np.zeros((len(word2id)))
    for i in range(len(word2id)):
        similar[i] = cos_similarity(word_vecs[word2id[query_word]], word_vecs[i])
    
    cnt = 0
    for i in (-1 * similar).argsort():
        if i == word2id[query_word]:
            continue
        print(f"{id2word[i]}, {similar[i]}")
        cnt += 1
        if cnt == top:
            return


def analogy(a, b, c, word2id, id2word, word_vecs:np.array, top=5, answer=None):
    # a-b = c-d => d = c - a + b
    a_vec, b_vec, c_vec = word_vecs[word2id[a]], word_vecs[word2id[b]], word_vecs[word2id[c]]
    d_vec = c_vec - a_vec + b_vec

    d_vec = d_vec / np.sqrt(np.sum(d_vec ** 2))  # normalize

    similarity = np.zeros((len(word2id)))
    for i in range(len(word2id)):
        similarity[i] = cos_similarity(d_vec, word_vecs[i])
    
    cnt = 0
    print(f"{a}-{b} == {c}-?:")
    for i in (-1 * similarity).argsort():  # return sorted index
        if i in (word2id[a], word2id[b], word2id[c]):
            continue
        print(f"? = {id2word[i]}, similarity = {similarity[i]}")
        cnt += 1
        if cnt == top:
            return
```

