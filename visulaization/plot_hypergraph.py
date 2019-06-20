import json
import hypernetx as hnx
import matplotlib.pyplot as plt
from hypernetx.drawing.rubber_band import draw

with open("../citation_dataset/citation_dataset.json") as graph:
    graph = json.load(graph)
print(graph)
d1 = {key: value for i, (key, value) in enumerate(graph.items()) if i % 100 == 0}
d2 = {key: value for i, (key, value) in enumerate(graph.items()) if i % 2 == 1}
H = hnx.Hypergraph(d1)


draw(H, ax=plt.subplot(121))
plt.show()
