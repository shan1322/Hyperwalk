from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import json


class SkipGram:
    def __init__(self):
        self.latent_dimension = 10
        self.max_length = 6
        self.vocab_size = 9

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
        skip_gram.fit(features, labels, verbose=2, batch_size=32, epochs=20, shuffle=True)
        weights = skip_gram.layers[0].get_weights()
        return weights

    def one_hot_encode(self, features):
        """

        :param features:graph nodes
        :return: one hot vectors
        """
        features = to_categorical(features, num_classes=9)
        return features

    def recover_embedding(self, features, labels):
        """

        :param features:graph nodes
        :param labels: labels
        :return: embeddings json
        """
        features_one_hot = skip_gram_obj.one_hot_encode(features)
        embedding = np.asarray(skip_gram_obj.train(feature, labels))
        embedding_json = {}
        for index in tqdm(range(len(features_one_hot))):
            individual_embedding = np.matmul(features_one_hot[index], embedding)
            individual_embedding = individual_embedding.reshape(individual_embedding.shape[1],
                                                                individual_embedding.shape[2])
            list_embedding = []
            for emb in individual_embedding:
                emb = emb.tolist()
                list_embedding.append(emb)
            for index_1 in range(len(feature[index])):
                if features[index][index_1] not in embedding_json.keys():
                    embedding_json[str(features[index][index_1])] = list_embedding[0]
        return embedding_json


feature, label = np.load("../toy_data/walk_dataset/data.npy",allow_pickle=True), np.load("../toy_data/walk_dataset/label.npy")
print(feature.shape)
skip_gram_obj = SkipGram()
json_emb = skip_gram_obj.recover_embedding(feature, label)
with open("../embeddings/node_embeddings.json", 'w') as node_embedding:
    json.dump(json_emb, node_embedding)
