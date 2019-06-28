import json
import hypernetx as hnx
import matplotlib.pyplot as plt
from hypernetx.drawing.rubber_band import draw

with open("../toy_data/iris_graph.json") as graph:
    graph = json.load(graph)

H = hnx.Hypergraph(graph)


draw(H)
plt.show()
