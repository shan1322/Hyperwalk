import json
import random
import numpy as np
from tqdm import tqdm
import pickle

with open("../toy_data/graph.json") as graph:
    graph = json.load(graph)


# with open("../citation_dataset/labels.pkl", 'rb') as file:
#   model = pickle.load(file)


class RandomWalk:
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
        print(self.inverse_map)

    def first_neighbour_edges(self, start_vertex):
        """
        :param:start vertex
        :return: dict containing first hop neighbours as key and walks to take in neighbours as value
        """
        neighbour_vertices = []
        neighbour_edges = []
        neighbour_edges_count = {}
        total_vertices = 0
        count_walk = 0

        vertex_to_edge = self.inverse_map
        for vertices in list(self.graph.values()):
            if start_vertex in vertices:
                neighbour_vertices.extend(vertices)
        for vertices in neighbour_vertices:
            neighbour_edges.extend(vertex_to_edge[vertices])
        neighbour_edges = set(neighbour_edges)
        for edges in neighbour_edges:
            neighbour_edges_count[edges] = len(self.graph[edges])
            total_vertices = total_vertices + len(self.graph[edges])
        for edges in neighbour_edges_count.keys():
            walk_per_edge = int((neighbour_edges_count[edges] / total_vertices) * self.walk_length)
            count_walk = count_walk + walk_per_edge
            neighbour_edges_count[edges] = walk_per_edge
            if walk_per_edge == 0:
                walk_per_edge = 1
        edge_list = list(neighbour_edges_count.keys())
        if count_walk != self.walk_length:
            missing_walk = self.walk_length - count_walk
            for walk_count in range(missing_walk):
                neighbour_edges_count[edge_list[walk_count]] = neighbour_edges_count[edge_list[walk_count]] + 1

        return list(neighbour_edges), neighbour_edges_count

    def single_walk(self, start_vertex):
        """
        :param:start vertex
        :return: vertices crossed in one random walk
        """
        walk = [start_vertex]
        neighbour_edges, neighbour_edges_count = self.first_neighbour_edges(start_vertex)
        count_vertex = []
        for edges in neighbour_edges:
            count_vertex.extend(self.graph[edges])

        while sum(list(neighbour_edges_count.values())) > 0:
            edges = random.choice(neighbour_edges)
            next_vertex = random.choice(self.graph[edges])
            if neighbour_edges_count[edges] > 0:
                walk.append(next_vertex)
                neighbour_edges_count[edges] = neighbour_edges_count[edges] - 1

        return walk

    def generate_walk_data_set(self, walks_per_vertex):
        """

        :param start_vertex
        :return: feature,labels
        """

        data = []
        label = []
        list_of_vertices = list(self.vertices)
        while len(list_of_vertices) > 0:
            start_vertex = random.choice(list_of_vertices)
            print(len(list_of_vertices))
            for iteration in range(walks_per_vertex):
                if iteration % 4 == 0:
                    data.append(self.single_walk(start_vertex))
                    label.append(0)
                else:
                    data.append(self.single_walk(start_vertex))
                    label.append(1)
            list_of_vertices.remove(start_vertex)
            print(list_of_vertices)
        return np.asarray(data), np.asarray(label)


walk = RandomWalk()
data, label = (walk.generate_walk_data_set(150))
print(data.shape)
np.save("../toy_data/walk_dataset/data.npy", data)
np.save("../toy_data/walk_dataset/label.npy", label)
