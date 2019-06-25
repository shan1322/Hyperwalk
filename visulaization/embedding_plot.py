from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import matplotlib.cm as cm

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
#for i in tqdm(temp):
 #   key.append(encoder.inverse_transform([int(i)])[0])
  #  for j in label:
   #     if encoder.inverse_transform([int(i)])[0] in label_dict[j]:
    #        labels.append(j)
    #val.append(emb[i])
#print(labels)
for i in tqdm(emb.values()):
    val.append(i)
X_embedded = PCA(n_components=50).fit_transform(val)
X_embedded = TSNE(n_components=2, verbose=2).fit_transform(X_embedded)
x, y = [], []
for i in tqdm(range(len(X_embedded))):
    x.append(X_embedded[i][0])
    y.append(X_embedded[i][1])
ax = plt.subplot2grid((5, 2), (0, 0))
ax1 = plt.subplot2grid((5, 2), (0, 1))
ax2= plt.subplot2grid((5, 2), (1, 0))
ax3 = plt.subplot2grid((5, 2), (1, 1))
ax4 = plt.subplot2grid((5, 2), (2, 0))
ax5 = plt.subplot2grid((5, 2), (2, 1))
ax6 = plt.subplot2grid((5, 2), (3, 0))
ax7 = plt.subplot2grid((5, 2), (3, 1))
ax8 = plt.subplot2grid((5, 2), (4, 0))
ax9 = plt.subplot2grid((5, 2), (4, 1))
ax.set_xlim([-200,200])
ax.set_ylim([-200,200])
ax1.set_xlim([-200,200])
ax1.set_ylim([-200,200])
ax2.set_xlim([-200,200])
ax2.set_ylim([-200,100])
ax3.set_xlim([-200,200])
ax3.set_ylim([-200,200])
ax4.set_xlim([-200,200])
ax4.set_ylim([-200,200])
ax5.set_xlim([-200,200])
ax5.set_ylim([-200,200])
ax6.set_xlim([-200,200])
ax6.set_ylim([-200,200])
ax7.set_xlim([-200,200])
ax7.set_ylim([-200,100])
ax8.set_xlim([-200,200])
ax8.set_ylim([-200,200])
ax9.set_xlim([-200,200])
ax9.set_ylim([-200,200])
for i in range(len(x)):
    print(i)
    if label.index(labels[i]) == 0:
        ax.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 1:
        ax1.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 2:
        ax2.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 3:
        ax3.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 4:
        ax4.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])

    if label.index(labels[i]) == 5:
        ax5.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 6:
        ax6.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 7:
        ax7.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 8:
        ax8.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
    if label.index(labels[i]) == 9:
        ax9.scatter(x[i], y[i], color=plot_schema[label.index(labels[i])])
# for i, txt in enumerate(key):
#   plt.annotate(txt, (x[i], y[i]))
plt.show()
