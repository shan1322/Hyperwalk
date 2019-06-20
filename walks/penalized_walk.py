import json
import hypernetx as hnx
import random
import numpy as np
import pickle
from tqdm import tqdm

with open("../toy_data/graph.json") as graph:
    graph = json.load(graph)


# with open("../citation_dataset/labels.pkl", 'rb') as file:
#   model = pickle.load(file)


class PenalizedWalk:
    def __init__(self):
        self.walk_length = 5
        self.graph = graph
        self.inverse_map = {}
        self.vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for vertex in tqdm(self.vertices):
            edges = []
            for edge in self.graph.keys():
                if vertex in self.graph[edge]:
                    edges.append(edge)
            self.inverse_map[vertex] = edges

    def neighbour_vertices(self, edge):
        neighbours = []
        for vertices in self.graph[edge]:
            neighbours.extend(self.inverse_map[vertices])
        neighbours = set(neighbours)
        neighbours.discard(edge)
        return list(neighbours)
 #   def single_walk(self):


penalized_walk = PenalizedWalk()
print(penalized_walk.neighbour_vertices('B'))
