import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def derivative_sigmoid_array(x):
    for i in range(len(x)):
        x[i]=derivative_sigmoid(x[i])
    return x