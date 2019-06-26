import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

with open("../embeddings/node_embeddings_2.json") as emb:
    emb = json.load(emb)
with open("../citation_dataset/citation_label.json") as labels:
    labels = json.load(labels)
features, classes = [], []
for emb_key in tqdm(emb.keys()):
    for labels_key in labels.keys():
        if emb_key in labels[labels_key]:
            features.append(emb[emb_key])
            classes.append(labels_key)
features, classes = np.array(features), np.array(classes)
features = features.reshape(features.shape[0], features.shape[2])

train_features, test_features, train_labels, test_labels = train_test_split(features, classes, test_size=0.2,
                                                                            random_state=42)
np.save("../citation_dataset/train_features.npy", train_features)
np.save("../citation_dataset/test_features.npy", test_features)
np.save("../citation_dataset/train_labels.npy", train_labels)
np.save("../citation_dataset/test_labels.npy", test_labels)
