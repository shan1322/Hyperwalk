import json
from tqdm import tqdm
import pickle
import numpy as np

with open("../citation_dataset/citation_dataset.json") as graph:
    graph = json.load(graph)

with open("../citation_dataset/labels.pkl", 'rb') as file:
    model = pickle.load(file)
inverse_map = {}
vertices = model.classes_

for vertex in tqdm(vertices):
    edges = []
    for edge in graph.keys():
        if vertex in graph[edge]:
            edges.append(edge)
    inverse_map[vertex] = edges


def first_hop_neighbour(edge_):
    start_vertex = graph[edge_][0]
    neighbour_vertices = []
    neighbour_edges = []
    edges_of_start_vertex = inverse_map[start_vertex]
    for edges in edges_of_start_vertex:
        neighbour_vertices.extend(graph[edges])
    neighbour_vertices = set(neighbour_vertices)
    for vertices in neighbour_vertices:
        neighbour_edges.extend(inverse_map[vertices])
    neighbour_edges = set(neighbour_edges)
    return list(neighbour_edges)


def median_neighbour_nodes():
    number_of_neighbours = []
    number_of_nodes = []
    for key in tqdm(graph.keys()):
        number_of_neighbours.append(len(first_hop_neighbour(key)))
        number_of_nodes.append(len(graph[key]))
    return np.percentile(number_of_neighbours, 50), number_of_neighbours, np.percentile(number_of_nodes,
                                                                                        50), number_of_nodes


print(median_neighbour_nodes()[2])
