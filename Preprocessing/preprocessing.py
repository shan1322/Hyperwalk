import json
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm


def convert_to_array(path):
    data = open(path, "r")
    hyper_graph_json = {}
    count = 0
    vertices = []
    for edges in data:
        edges = edges.strip("\n")
        edges = edges.split(' ')
        vertices.extend(edges)
        hyper_graph_json["A" + str(count)] = edges
        count = count + 1
    return hyper_graph_json, vertices


def encode(vertices):
    label_encoder = LabelEncoder()
    label_encoder.fit(vertices)
    with open("../citation_dataset/labels.pkl", 'wb') as file:
        pickle.dump(label_encoder, file)


def encode_json(graph):
    with open("../citation_dataset/labels.pkl", 'rb') as file:
        model = pickle.load(file)
    for edges in tqdm(graph.keys()):
        graph[edges] = model.transform(graph[edges]).tolist()
    return graph


def label(path):
    data = open(path, "r")
    label_graph_json = {'1': list(), '2': list(), '3': list(), '4': list(), '5': list(), '6': list(), '7': list(),
                        '8': list(), "9": list(), "10": list()}
    for edges in tqdm(data):
        edges = edges.strip("\n")
        edges = edges.split(' ')
        x = label_graph_json[edges[1]]
        x.append(edges[0])
        label_graph_json[edges[1]] = x
    return label_graph_json


with open("../citation_dataset/citation_label.json", 'w') as graph:
    json.dump(label("../citation_dataset/paperid_classid.txt")
              , graph)
# hyper_graph, vertices = convert_to_array("../citation_dataset/hyperedges.txt")

# with open("../citation_dataset/citation_dataset.json", 'w') as graph:
#   json.dump(hyper_graph, graph)
# encode(vertices)

# with open("../citation_dataset/citation_dataset_encoded.json", 'w') as graph:
#    json.dump(encode_json(hyper_graph), graph)
