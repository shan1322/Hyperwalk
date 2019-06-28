from sklearn.manifold import TSNE
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
label = iris['target']
with open("../embeddings/node_embeddings_iris.json") as emb:
    emb = json.load(emb)
key, val = [], []
for i in tqdm(emb.keys()):
    key.append(i)
    val.append(emb[i])
val = np.array(val)
val = val.reshape(val.shape[0], val.shape[2])
X_embedded = TSNE(n_components=2, verbose=2,perplexity=50).fit_transform(val)
x, y = [], []
for i in tqdm(range(len(X_embedded))):
    if label[i] == 0:
        plt.scatter(X_embedded[i][0], X_embedded[i][1],color='r')
    if label[i] == 1:
        plt.scatter(X_embedded[i][0], X_embedded[i][1],color= 'g')
    if label[i] == 2:
        plt.scatter(X_embedded[i][0], X_embedded[i][1], color='b')
# for i, txt in enumerate(key):
#   plt.annotate(txt, (x[i], y[i]))
plt.show()
