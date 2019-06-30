import json
import random
from tqdm import tqdm
import numpy as np

with open("../citation_dataset/clique_transformed.json") as graph:
    graph = json.load(graph)
modified_graph = {}
for i in tqdm(graph.keys()):
    if len(graph[i]) > 0:
        modified_graph[i] = graph[i]


class FirstHopCliqueWalk:
    def __init__(self):
        self.graph = modified_graph
        self.graph_0 = graph
        self.walk_length = 396

    def first_hop_neighbour_vertices(self, star_vertex):
        adjacency_vertices = self.graph[str(star_vertex)]
        neighbour_vertices = []
        neighbour_vertices.extend(adjacency_vertices)
        for vertices in adjacency_vertices:
            neighbour_vertices.extend(self.graph_0[str(vertices)])
        neighbour_vertices = list(set(neighbour_vertices))
        return neighbour_vertices

    def single_walk(self, start_vertex):
        walk_path = [start_vertex]
        while len(walk_path) < self.walk_length:
            neighbour_vertices = self.graph[str(walk_path[-1])]
            next_vertex = random.choice(neighbour_vertices)
            if next_vertex in self.first_hop_neighbour_vertices(walk_path[-1]) and next_vertex != walk_path[-1]:
                walk_path.append(next_vertex)
        return walk_path

    def generate_walk_data_set(self, walks_per_vertex):
        """

        :param start_vertex
        :return: feature,labels
        """

        data = []
        label = []
        list_of_vertices = [i for i in range(21375)]
        for start_vertex in tqdm(list_of_vertices):
            for iteration in range(walks_per_vertex):
                if iteration % 10 == 0:
                    walk_path = self.single_walk(start_vertex)
                    data.append(walk_path)
                    label.append(1)
                    walk_path = walk_path[0:len(walk_path) - 1]
                    false_vertex = random.choice(list_of_vertices)
                    if false_vertex not in self.graph[str(start_vertex)]:
                        walk_path.append(false_vertex)
                        data.append(walk_path)
                        label.append(0)

                else:
                    walk_path = self.single_walk(start_vertex)
                    data.append(walk_path)
                    label.append(1)
        return np.asarray(data), np.asarray(label)


walk = FirstHopCliqueWalk()
data, label = (walk.generate_walk_data_set(10))
print(data.shape)
np.save("../toy_data/walk_dataset/label_clique.npy", label)
np.save("../toy_data/walk_dataset/data_clique.npy", data)
