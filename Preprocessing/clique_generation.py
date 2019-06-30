import json
from tqdm import tqdm

with open("../citation_dataset/citation_dataset_encoded.json") as graph:
    graph = json.load(graph)


def clique_generation(hyper_graph):
    sparse_adjacency = {}
    for index in tqdm(range(21375)):
        neighbours = []
        for edges in hyper_graph.keys():
            neighbours_vertices = hyper_graph[edges]
            if index in neighbours_vertices:
                neighbours_vertices.remove(index)
                neighbours.extend(neighbours_vertices)
        sparse_adjacency[str(index)] = list(set(neighbours))
    return sparse_adjacency


out = clique_generation(graph)
print(out)
with open("../citation_dataset/clique_transformed.json", 'w') as graph:
    json.dump(out, graph)
