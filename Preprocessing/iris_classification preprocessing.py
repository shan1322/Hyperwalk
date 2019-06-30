import json
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

label = np.load("../toy_data/abone.npy")
label=label[:,label.shape[1]-1]
label=label-1
label=label.astype(int)
with open("../embeddings/node_embeddings_abone.json") as emb:
    emb = json.load(emb)
features, classes = [], []
for index in tqdm(emb.keys()):
    features.append(emb[index])
    classes.append(label[int(index)])
features = np.asarray(features)
features = features.reshape(features.shape[0], features.shape[2])
train_features, test_features, train_labels, test_labels = train_test_split(features, classes, test_size=0.1,
                                                                            random_state=42)
np.save("../citation_dataset/train_features.npy", train_features)
np.save("../citation_dataset/test_features.npy", test_features)
np.save("../citation_dataset/train_labels.npy", train_labels)
np.save("../citation_dataset/test_labels.npy", test_labels)
print(train_features.shape)
print(set(train_labels))