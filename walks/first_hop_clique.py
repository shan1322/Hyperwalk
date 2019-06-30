import json
import random
from tqdm import tqdm

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
        self.walk_length = 6

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


walk = FirstHopCliqueWalk()
print(walk.single_walk(3))
