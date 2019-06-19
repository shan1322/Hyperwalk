import matplotlib.pyplot as plt
import numpy as np
features = np.load("../toy_data/walk_dataset/data.npy", allow_pickle=True)


def count_frequency(start_vertex, feature):
    walk_start_vertex = []
    for walk in feature:
        if start_vertex == walk[0]:
            walk_start_vertex.extend(walk)

    frequency_dict = {vertex: walk_start_vertex.count(vertex) for vertex in walk_start_vertex}
    return frequency_dict


print(count_frequency(1, features))
