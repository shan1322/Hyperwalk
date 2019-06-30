from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import json



class SkipGram:
    def __init__(self):
        self.latent_dimension = 64
        self.max_length = 100
        self.vocab_size = 4176

    def skip_gram_model(self):
        """

        :return: skip gram models
        """

        model = Sequential()
        model.add(Embedding(self.vocab_size, self.latent_dimension, input_length=self.max_length))
        model.add(Flatten())
        model.add(Dense(4176, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        return model

    def train(self, features, labels):
        """

        :param features:vertices
        :param labels: label
        :return: weights of embedding layer
        """

        skip_gram = self.skip_gram_model()
        skip_gram.fit(features, labels, verbose=2, batch_size=100, epochs=10, shuffle=True)
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
        for index in tqdm(range(4176)):
            feature_one_hot = self.one_hot_encode([index])
            individual_embedding = np.matmul(feature_one_hot, embedding)
            individual_embedding = individual_embedding.reshape(individual_embedding.shape[1],
                                                                individual_embedding.shape[2])

            embedding_json[str(index)] = individual_embedding.tolist()
        return embedding_json


features = np.load("../toy_data/walk_dataset/data-iris.npy")
label = features[:, [9]]
features = np.delete(features, 9, 1)
skip_gram_obj = SkipGram()
json_emb = skip_gram_obj.recover_embedding(features, label)
with open("../embeddings/node_embeddings_abone.json", 'w') as node_embedding:
    json.dump(json_emb, node_embedding)
