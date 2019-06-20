import json
import hypernetx as hnx
import matplotlib.pyplot as plt
from hypernetx.drawing.rubber_band import draw
with open("../citation_dataset/citation_dataset.json") as graph:
    graph = json.load(graph)
print(graph)
H = hnx.Hypergraph(graph)

plt.figure(figsize=(20,10))
draw(H, ax=plt.subplot(121))
plt.show()
