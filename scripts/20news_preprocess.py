import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np

from lib.utils import Text20News


remove = ('headers', 'footers', 'quotes')  # (), ('headers') or ('headers','footers','quotes')
train = Text20News(data_home='data', subset='train', remove=remove)
print(train.show_document(1)[:400])
train.clean_text(num='substitute')
train.vectorize(stop_words='english')
print(train.show_document(1)[:400])
train.data_info(True)
wc = train.remove_short_documents(nwords=20, vocab='full')
train.data_info()
print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
plt.figure(figsize=(17,5))
plt.semilogy(wc, '.')

def remove_encoded_images(dataset, freq=1e3):
    widx = train.vocab.index('ax')
    wc = train.data[:,widx].toarray().squeeze()
    idx = np.argwhere(wc < freq).squeeze()
    dataset.keep_documents(idx)
    return wc
wc = remove_encoded_images(train)
train.data_info()

# train.embed(os.path.join('data', 'GoogleNews-vectors-negative300.bin'))
train.embed()
train.data_info()
freq = train.keep_top_words(1000, 20)
train.data_info()
train.show_document(1)
plt.figure(figsize=(17,5))
plt.semilogy(freq)

# Remove documents whose signal would be the zero vector.
wc = train.remove_short_documents(nwords=5, vocab='selected')
train.data_info(True)
train.normalize(norm='l1')
train.show_document(1)

if not os.path.exists('data/20news'):
    os.mkdir('data/20news')
sparse.save_npz('data/20news/train_data.npz', train.data)
np.save('data/20news/train_labels.npy', train.labels)
np.save('data/20news/embeddings.npy', train.embeddings)
with open('data/20news/class_names.txt', 'w') as fw:
    fw.writelines([c+'\n' for c in train.class_names])

test = Text20News(data_home='data', subset='test', remove=remove)
test.clean_text(num='substitute')
test.vectorize(vocabulary=train.vocab)
test.data_info()
wc = test.remove_short_documents(nwords=5, vocab='selected')
print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
test.data_info(True)
test.normalize(norm='l1')

sparse.save_npz('data/20news/test_data.npz', test.data)
np.save('data/20news/test_labels.npy', test.labels)
