import json
from tqdm import tqdm
import random
import numpy as np
import pickle

with open("../toy_data/graph.json") as graph:
    graph = json.load(graph)


class FirstHopWalkBudget:
    def __init__(self):
        self.walk_length = 7
        self.graph = graph
        self.inverse_map = {}
        self.vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for vertex in tqdm(self.vertices):
            edges = []
            for edge in self.graph.keys():
                if vertex in self.graph[edge]:
                    edges.append(edge)
            self.inverse_map[vertex] = edges

    def firs_hop_neighbour(self, start_vertex):
        neighbour_vertices = []
        neighbour_edges = []
        neighbour_walk_count = {}
        edges_of_start_vertex = self.inverse_map[start_vertex]
        for edges in edges_of_start_vertex:
            neighbour_vertices.extend(self.graph[edges])
        neighbour_vertices = set(neighbour_vertices)
        for vertices in neighbour_vertices:
            neighbour_edges.extend(self.inverse_map[vertices])
        neighbour_edges = set(neighbour_edges)
        walk_length = int(self.walk_length / len(neighbour_edges)) * len(neighbour_edges)
        count = self.walk_length - walk_length
        for edges in neighbour_edges:
            if count > 0:
                neighbour_walk_count[edges] = int(self.walk_length / len(neighbour_edges)) + 1
                count = count - 1
            elif count == 0:
                neighbour_walk_count[edges] = int(self.walk_length / len(neighbour_edges))

        return list(neighbour_edges), neighbour_walk_count

    def single_walk(self, start_vertex):
        current_walk_length = self.walk_length
        first_hop_list, first_hop_dict = self.firs_hop_neighbour(start_vertex)
        walk_path = [start_vertex]
        while len(walk_path) <= current_walk_length:
            neighbours_in_first_hop = []
            first_hop_neighbour_list = self.firs_hop_neighbour(walk_path[-1])
            for edge in first_hop_neighbour_list:
                if edge in first_hop_list:
                    neighbours_in_first_hop.append(edge)
            next_edge = random.choice(neighbours_in_first_hop)
            next_vertex = random.choice(self.graph[next_edge])
            walk_path.append(next_vertex)
        return walk_path


budget = FirstHopWalkBudget()
print(budget.firs_hop_neighbour(1))
