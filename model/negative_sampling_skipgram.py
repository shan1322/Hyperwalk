from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import json
import pickle

with open("../citation_dataset/citation_dataset.json") as graph:
    graph = json.load(graph)

with open("../citation_dataset/labels.pkl", 'rb') as file:
    model = pickle.load(file)


class SkipGram:
    def __init__(self):
        self.latent_dimension = 64
        self.label_encoder = model
        self.max_length = 396
        self.vocab_size = len(self.label_encoder.classes_)

    def skip_gram_model(self):
        """

        :return: skip gram models
        """

        model = Sequential()
        model.add(Embedding(self.vocab_size, self.latent_dimension, input_length=self.max_length))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        return model

    def train(self, features, labels):
        """

        :param features:vertices
        :param labels: label
        :return: weights of embedding layer
        """

        skip_gram = self.skip_gram_model()
        skip_gram.fit(features, labels, verbose=2, batch_size=1000, epochs=20, shuffle=True)
        weights = skip_gram.layers[0].get_weights()
        return weights

    def one_hot_encode(self, features):
        """

        :param features:graph nodes
        :return: one hot vectors
        """
        features = to_categorical(features, num_classes=self.vocab_size)
        return features

    def recover_embedding(self, feature, labels):
        """

        :param features:graph nodes
        :param labels: labels
        :return: embeddings json
        """
        embedding = np.asarray(self.train(feature, labels))
        embedding_json = {}
        for index in tqdm(range(21375)):
            feature_one_hot = self.one_hot_encode([index])
            individual_embedding = np.matmul(feature_one_hot, embedding)
            individual_embedding = individual_embedding.reshape(individual_embedding.shape[1],
                                                                individual_embedding.shape[2])

            embedding_json[self.label_encoder.inverse_transform([index])[0]] = individual_embedding.tolist()
        return embedding_json


features, label = np.load("../toy_data/walk_dataset/data_clique.npy", allow_pickle=True), np.load(
    "../toy_data/walk_dataset/label_clique.npy")
#label = features[:, [6]]
#features = np.delete(features, 6, 1)
# temp=[]
# for i in tqdm(features):
#   temp.append(model.transform(i))
# np.save("../toy_data/walk_dataset/data_encoded_16.npy",temp)
skip_gram_obj = SkipGram()
json_emb = skip_gram_obj.recover_embedding(features, label)
with open("../embeddings/node_embeddings_clique.json", 'w') as node_embedding:
    json.dump(json_emb, node_embedding)
