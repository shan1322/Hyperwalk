import json
from tqdm import tqdm
import random
import numpy as np
import pickle

with open("../citation_dataset/citation_dataset.json") as graph:
    graph = json.load(graph)

with open("../citation_dataset/labels.pkl", 'rb') as file:
    model = pickle.load(file)


class FirstHopWalk:
    def __init__(self):
        self.walk_length = 30
        self.graph = graph
        self.inverse_map = {}
        self.vertices = model.classes_
        for vertex in tqdm(self.vertices):
            edges = []
            for edge in self.graph.keys():
                if vertex in self.graph[edge]:
                    edges.append(edge)
            self.inverse_map[vertex] = edges

    def firs_hop_neighbour(self, start_vertex):
        neighbour_vertices = []
        neighbour_edges = []
        edges_of_start_vertex = self.inverse_map[start_vertex]
        for edges in edges_of_start_vertex:
            neighbour_vertices.extend(self.graph[edges])
        neighbour_vertices = set(neighbour_vertices)
        for vertices in neighbour_vertices:
            neighbour_edges.extend(self.inverse_map[vertices])
        neighbour_edges = set(neighbour_edges)
        return list(neighbour_edges)

    def single_walk(self, start_vertex):
        current_walk_length = self.walk_length
        first_hop_list=self.firs_hop_neighbour(start_vertex)
        walk_path = [start_vertex]
        while len(walk_path) <= current_walk_length:
            neighbours_in_first_hop = []
            first_hop_neighbour_list=self.firs_hop_neighbour(walk_path[-1])
            for edge in first_hop_neighbour_list:
                if edge in first_hop_list:
                    neighbours_in_first_hop.append(edge)
            next_edge = random.choice(neighbours_in_first_hop)
            next_vertex = random.choice(self.graph[next_edge])

            walk_path.append(next_vertex)
        return walk_path

    def generate_walk_data_set(self, walks_per_vertex):
        """

        :param start_vertex
        :return: feature,labels
        """

        data = []
        label = []
        list_of_vertices = self.vertices
        for start_vertex in tqdm(list_of_vertices):
            for iteration in range(walks_per_vertex):
                if iteration % 4 == 0:
                    data.append(self.single_walk(start_vertex))
                    label.append(0)
                else:
                    data.append(self.single_walk(start_vertex))
                    label.append(1)
        return np.asarray(data), np.asarray(label)


first_hop = FirstHopWalk()
data, label = (first_hop.generate_walk_data_set(10))

np.save("../toy_data/walk_dataset/label.npy", label)
np.save("../toy_data/walk_dataset/data.npy", data)
