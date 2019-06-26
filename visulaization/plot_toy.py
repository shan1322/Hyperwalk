from sklearn.manifold import TSNE
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
with open("../embeddings/node_embeddings_2.json") as emb:
    emb = json.load(emb)
key, val = [], []
for i in tqdm(emb.keys()):
    key.append(i)
    val.append(emb[i])
val=np.array(val)
val=val.reshape(val.shape[0],val.shape[2])
X_embedded = TSNE(n_components=2, verbose=2).fit_transform(val)
x, y = [], []
for i in tqdm(range(len(X_embedded))):
    x.append(X_embedded[i][0])
    y.append(X_embedded[i][1])
plt.scatter(x,y)
#for i, txt in enumerate(key):
 #   plt.annotate(txt, (x[i], y[i]))
plt.show()