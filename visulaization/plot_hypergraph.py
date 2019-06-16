import json
import hypernetx as hnx
import matplotlib.pyplot as plt
from hypernetx.drawing.rubber_band import draw
with open("../toy_data/graph.json") as graph:
    graph = json.load(graph)
H = hnx.Hypergraph(graph)
plt.figure(figsize=(16, 8))
draw(H, ax=plt.subplot(121))
plt.title('Dual');
plt.show()