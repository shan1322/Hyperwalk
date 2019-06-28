from sklearn import datasets
import numpy as np
import json

iris = datasets.load_iris()


def quartile_calculation(features):
    quartile_one = np.percentile(features, 20)
    quartile_two = np.percentile(features, 40)
    quartile_three = np.percentile(features, 60)
    quartile_four = np.percentile(features, 80)
    return quartile_one, quartile_two, quartile_three, quartile_four


def make_hyper_edges(features):
    q_1, q_2, q_3,q_4 = (quartile_calculation(features))
    f_1, f_2, f_3, f_4,f_5 = [], [], [], [],[]
    for index in range(len(features)):
        if features[index] < q_1:
            f_1.append(index)
        elif q_1 <= features[index] < q_2:
            f_2.append(index)
        elif q_2 <= features[index] < q_3:
            f_3.append(index)
        elif q_3 <= features[index] < q_4:
            f_4.append(index)
        elif features[index] >= q_4:
            f_5.append(index)
    return f_1, f_2, f_3, f_4,f_5


def make_hyper_graph(data_set):
    hyper_graph = {}
    for columns in range(data_set.shape[1]):
        f_1, f_2, f_3, f_4,f_5 = make_hyper_edges(data_set[:, columns])
        hyper_graph['A' + str(columns + 1) + "f1"] = f_1
        hyper_graph['A' + str(columns + 1) + "f2"] = f_2
        hyper_graph['A' + str(columns + 1) + "f3"] = f_3
        hyper_graph['A' + str(columns + 1) + "f4"] = f_4
        hyper_graph['A' + str(columns + 1) + "f5"] = f_5

    return hyper_graph


with open("../toy_data/iris_graph.json", 'w') as graph:
    json.dump(make_hyper_graph(iris['data']), graph)
