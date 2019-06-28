from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np

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
val = np.array(val)
val = val.reshape(val.shape[0], val.shape[2])
X_embedded = PCA(n_components=50).fit_transform(val)
X_embedded = PCA(n_components=40).fit_transform(X_embedded)
X_embedded = PCA(n_components=30).fit_transform(X_embedded)
X_embedded = PCA(n_components=20).fit_transform(X_embedded)
X_embedded = PCA(n_components=10).fit_transform(X_embedded)
X_embedded = PCA(n_components=5).fit_transform(X_embedded)
X_embedded = TSNE(n_components=2, verbose=2, metric="cosine").fit_transform(X_embedded)
x, y = [], []
for i in tqdm(range(len(X_embedded))):
    x.append(X_embedded[i][0])
    y.append(X_embedded[i][1])

print(labels)
xm0, ym0, xm1, ym1, xm2, ym2, xm3, ym3, xm4, ym4, xm5, ym5, xm6, ym6, xm7, ym7, xm8, ym8, xm9, ym9 = [0, 0, 0, 0, 0, 0,
                                                                                                      0, 0, 0, 0, 0, 0,
                                                                                                      0, 0, 0, 0, 0, 0,
                                                                                                      0, 0]
for i in range(len(x)):
    print(labels[i])
    print(label.index(labels[i]))
    if label.index(labels[i]) == 0:
        # ax.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm0 = xm0 + x[i]
        ym0 = ym0 + y[i]
    if label.index(labels[i]) == 1:
        # ax1.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm1 = xm1 + x[i]
        ym1 = ym1 + y[i]
    if label.index(labels[i]) == 2:
        # ax2.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm2 = xm2 + x[i]
        ym2 = ym2 + y[i]
    if label.index(labels[i]) == 3:
        # ax3.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm3 = xm3 + x[i]
        ym3 = ym3 + y[i]
    if label.index(labels[i]) == 4:
        # ax4.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm4 = xm4 + x[i]
        ym4 = ym4 + y[i]
    if label.index(labels[i]) == 5:
        # ax5.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm5 = xm5 + x[i]
        ym5 = ym5 + y[i]
    if label.index(labels[i]) == 6:
        # ax6.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm6 = xm6 + x[i]
        ym6 = ym6 + y[i]
    if label.index(labels[i]) == 7:
        # ax7.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm7 = xm7 + x[i]
        ym7 = ym7 + y[i]
    if label.index(labels[i]) == 8:
        # ax8.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm8 = xm8 + x[i]
        ym8 = ym8 + y[i]
    if label.index(labels[i]) == 9:
        # ax9.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
        xm9 = xm9 + x[i]
        ym9 = ym9 + y[i]

mean_x = [xm0 / len(x), xm1 / len(x), xm2 / len(x), xm3 / len(x), xm4 / len(x), xm5 / len(x), xm6 / len(x), xm7 / len(x),
         xm8 / len(x), xm9 / len(x)]
mean_y = [ym0 / len(x), ym1 / len(x), ym2 / len(x), ym3 / len(x), ym4 / len(x), ym5 / len(x), ym6 / len(x), ym7 / len(x),
         ym8 / len(x), ym9 / len(x)]

plt.scatter(mean_x, mean_y)
#for i, txt in enumerate(mean_x):
#    plt.annotate(i, (x[i], y[i]))
plt.show()
