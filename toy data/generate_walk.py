import json
import hypernetx as hnx
import secrets

with open("../toy data/graph.json") as graph:
    graph = json.load(graph)


class RandomWalk:
    def __init__(self):
        self.walk_length = 6
        self.graph = graph
        self.hyper_graph = hnx.Hypergraph(graph)
        self.start_vertex = int(input())

    def first_neighbour_edges(self):
        neighbour_vertices = []
        neighbour_edges = []
        neighbour_edges_count = {}
        total_vertices = 0
        count_walk = 0

        vertex_to_edge = dict(self.hyper_graph.dual().edges.incidence_dict)
        for vertices in list(self.graph.values()):
            if self.start_vertex in vertices:
                neighbour_vertices.extend(vertices)
        for vertices in neighbour_vertices:
            neighbour_edges.extend(vertex_to_edge[vertices])
        neighbour_edges = set(neighbour_edges)
        for edges in neighbour_edges:
            neighbour_edges_count[edges] = len(self.graph[edges])
            total_vertices = total_vertices + len(self.graph[edges])
        for edges in neighbour_edges_count.keys():
            walk_per_edge = round((neighbour_edges_count[edges] / total_vertices) * self.walk_length)
            if walk_per_edge == 0:
                walk_per_edge = 1
            count_walk = count_walk + walk_per_edge
            neighbour_edges_count[edges] = walk_per_edge
        if count_walk != self.walk_length:
            print("error in walk")
        return list(neighbour_edges), neighbour_edges_count

    def single_walk(self):
        walk = [self.start_vertex]
        neighbour_edges, neighbour_edges_count = self.first_neighbour_edges()
        edge_completed = 0
        while edge_completed <= len(neighbour_edges):
            next_edge = secrets.choice(neighbour_edges)
            next_vertex = secrets.choice(self.graph[next_edge])
            if neighbour_edges_count[next_edge] != 0 and next_vertex not in walk:
                walk.append(next_vertex)
                neighbour_edges_count[next_edge] = neighbour_edges_count[next_edge] - 1
            elif neighbour_edges_count[next_edge] == 0:
                edge_completed = edge_completed + 1
        return walk


walk = RandomWalk()
print(walk.single_walk())
