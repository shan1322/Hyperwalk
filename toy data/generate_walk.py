import json
import hypernetx as hnx

with open("../toy data/graph.json") as graph:
    graph = json.load(graph)


class RandomWalk:
    def __init__(self):
        self.walk_length = 6
        self.graph = graph
        self.hyper_graph = hnx.Hypergraph(graph)

    def first_neighbour_edges(self, start_vertex):
        neighbour_vertices = []
        neighbour_edges = []
        vertex_to_edge = dict(self.hyper_graph.dual().edges.incidence_dict)
        for vertices in list(self.graph.values()):
            if start_vertex in vertices:
                neighbour_vertices.extend(vertices)
        for vertices in neighbour_vertices:
            neighbour_edges.extend(vertex_to_edge[vertices])
        return list(set(neighbour_edges))


walk = RandomWalk()
print(walk.first_neighbour_edges(1))
