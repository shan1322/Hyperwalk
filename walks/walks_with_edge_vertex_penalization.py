import json
import hypernetx as hnx
import random
import numpy as np
import pickle

with open("../toy_data/graph.json") as graph:
    graph = json.load(graph)
with open("../citation_dataset/labels.pkl", 'rb') as file:
    model = pickle.load(file)


class RandomPenalizationWalk:
    def __init__(self):
        self.graph = graph
        self.walk_length = 10
        self.hyper_graph = hnx.Hypergraph(graph)

        self.inverse_map = dict(self.hyper_graph.dual().edges.incidence_dict)
        self.vertices = []
        for vertices_list in self.graph.values():
            self.vertices.extend(vertices_list)
        self.vertices = set(self.vertices)

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

        vertex_to_edge = dict(self.hyper_graph.dual().edges.incidence_dict)
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

    def connecting_vertex(self, edge_1, edge_2):
        all_vertices = []
        common_vertices = []
        all_vertices.extend(self.graph[edge_1])
        all_vertices.extend(self.graph[edge_2])
        all_vertices = list(set(all_vertices))
        for vertices in all_vertices:
            if vertices in self.graph[edge_1] and vertices in self.graph[edge_2]:
                common_vertices.append(vertices)
        return common_vertices

    def single_walk(self, start_vertex):
        """
                :param:start vertex
                :return: vertices crossed in one random walk
                """
        walk = [start_vertex]
        neighbour_edges, neighbour_edges_count = self.first_neighbour_edges(start_vertex)
        while sum(list(neighbour_edges_count.values())) > 0:
            edges_to_be_covered = []
            for edges in self.inverse_map[walk[-1]]:
                for vertices in self.graph[edges]:
                    edges_to_be_covered.extend(self.inverse_map[vertices])
            edges = random.choice(list(set(edges_to_be_covered)))
            if edges in neighbour_edges:
                if neighbour_edges_count[edges] > 0:
                    next_vertex = random.choice(self.graph[edges])
                    walk.append(next_vertex)
                    neighbour_edges_count[edges] = neighbour_edges_count[edges] - 1
                    print(neighbour_edges_count, next_vertex)
                    print(walk)
            if edges not in neighbour_edges:
                neighbour_edges_count[edges] = 0
        return walk

    def generate_walk_data_set(self, walks_per_vertex):
        """

        :param start_vertex
        :return: feature,labels
        """

        data = []
        label = []
        list_of_vertices = list(self.vertices)
        start_vertex = random.choice(list_of_vertices)
        list_of_vertices.remove(start_vertex)
        while len(list_of_vertices) > 0:
            start_vertex = random.choice(list_of_vertices)

            for iteration in range(walks_per_vertex):
                if iteration % 3 == 0:
                    data.append(self.single_walk(start_vertex))
                    # print(self.single_walk(start_vertex))
                    label.append(0)
                else:
                    data.append(self.single_walk(start_vertex))
                    print(self.single_walk(start_vertex))
                    label.append(1)
            list_of_vertices.remove(start_vertex)
        return np.asarray(data), np.asarray(label)


random_walk = RandomPenalizationWalk()
# print(random_walk.single_walk(2))
data, label = (random_walk.generate_walk_data_set(1))
print(data.shape)
np.save("../toy_data/walk_dataset/data.npy", data)
np.save("../toy_data/walk_dataset/label.npy", label)
