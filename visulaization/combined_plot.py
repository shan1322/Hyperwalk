from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

with open("../embeddings/node_embeddings_2.json") as emb:
    emb = json.load(emb)
with open("../citation_dataset/labels.pkl", 'rb') as file:
    encoder = pickle.load(file)
with open("../citation_dataset/citation_label.json") as label:
    label_dict = json.load(label)
reduced_embedding = {}
label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
plot_schema = ["b", "g", "r", "c", "m", "y", "k", 'w', '#FFD700', '#FF00FF', '#C71585']
labels = []
key, val = [], []
temp = list(emb.keys())
print(encoder.classes_)
for i in tqdm(temp):
    if i not in label_dict['1']:
        key.append(i)
        for j in label:
            if i in label_dict[j]:
                labels.append(j)
        val.append(emb[i])
val = np.array(val)
val = val.reshape(val.shape[0], val.shape[2])

X_embedded = TSNE(n_components=2, verbose=2, metric="cosine", perplexity=5,n_iter=250).fit_transform(
    val)
x, y, z = [], [], []
for i in tqdm(range(len(X_embedded))):
    x.append(X_embedded[i][0] * 100000)
    y.append(X_embedded[i][1] * 100000)

count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9, count_10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
fig = plt.figure()
plt.figure(figsize=(20, 20))

for i in range(len(x)):
    print(label.index(labels[i]))
    plt.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])

# for i, txt in enumerate(key):
#   plt.annotate(txt, (x[i], y[i]))
plt.show()
