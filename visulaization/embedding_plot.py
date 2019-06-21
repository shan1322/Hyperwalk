from sklearn.manifold import TSNE
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
with open("../embeddings/node_embeddings.json") as emb:
    emb = json.load(emb)
print(emb)
reduced_embedding = {}
key, val = [], []
for i in emb.keys():
    key.append(i)
    val.append(emb[i])
X_embedded = TSNE(n_components=2,verbose=True).fit_transform(val)

x, y = [], []
for i in tqdm(range(len(X_embedded))):
    x.append(X_embedded[i][0])
    y.append(X_embedded[i][1])
fig = plt.scatter(x, y)

#for i, txt in enumerate(key):
 #   plt.annotate(txt, (x[i], y[i]))

plt.show()
