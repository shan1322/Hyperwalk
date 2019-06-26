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
    key.append(i)
    for j in label:
        if i in label_dict[j]:
            labels.append(j)
    val.append(emb[i])
val=np.array(val)
val = val.reshape(val.shape[0], val.shape[2])
print(set(labels))

X_embedded = PCA(n_components=10).fit_transform(val)
X_embedded = TSNE(n_components=3, verbose=2,n_iter=250,metric="cosine").fit_transform(X_embedded)
x, y,z = [], [],[]
for i in tqdm(range(len(X_embedded))):
    x.append(X_embedded[i][0])
    y.append(X_embedded[i][1])
    z.append(X_embedded[i][2])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(x)):
        ax.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])

# for i, txt in enumerate(key):
#   plt.annotate(txt, (x[i], y[i]))
plt.show()
