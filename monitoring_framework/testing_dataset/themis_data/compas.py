import numpy as np
import sys
import os
sys.path.append("../")

def compas_data(causal=True):
    """
    Prepare the data of dataset Compas
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0
    script_dir = os.path.dirname(__file__)
    if causal:
        read_file = "../datasets/causal_compas_datafile.csv"
    else:
        read_file = "../datasets/group_compas_datafile.csv"
    with open(os.path.join(script_dir,read_file), "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1]
            X.append(L)
    X = np.array(X, dtype=float)

    input_shape = (None, 16)
    nb_classes = 2

    return X, input_shape, nb_classes
