import json
import hypernetx as hnx
import random
import numpy as np

with open("../toy_data/graph.json") as graph:
    graph = json.load(graph)


class RandomWalk:
    def __init__(self):
        self.walk_length = 5
        self.graph = graph
        self.hyper_graph = hnx.Hypergraph(graph)
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
        try:
            count_vertex[self.walk_length]
        except IndexError:
            print("walk size exceeds neighbour vertices")
        for edges in neighbour_edges:
            while neighbour_edges_count[edges] > 0:
                # print(edges, neighbour_edges_count[edges])
                next_vertex = random.choice(self.graph[edges])
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
        start_vertex = random.choice(list_of_vertices)
        list_of_vertices.remove(start_vertex)
        while len(list_of_vertices) > 0:
            start_vertex = random.choice(list_of_vertices)
            # print(len(list_of_vertices))

            for iteration in range(walks_per_vertex):
                if iteration % 3 == 0:
                    data.append(self.single_walk(start_vertex))
                    print(start_vertex)
                    label.append(0)
                else:
                    print(start_vertex)
                    data.append(self.single_walk(start_vertex))
                    label.append(1)
            list_of_vertices.remove(start_vertex)
        return np.asarray(data), np.asarray(label)


walk = RandomWalk()
#print(walk.single_walk(0))
data, label = (walk.generate_walk_data_set(100))
np.save("walk_dataset/data.npy", data)
np.save("walk_dataset/label.npy", label)
