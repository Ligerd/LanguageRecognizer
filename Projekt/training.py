from predict import predict
import numpy as np
from maths import derivative_sigmoid
etta=0.001
def trainingWeights(treningData,answer,numberiteration,weights=None):
    wymiar = treningData[0].size
    if weights is None:
        np.random.seed(1)
        weights = np.array([2 * np.random.random((wymiar, wymiar)) - 1, 2 * np.random.random((int(wymiar / 2), wymiar)) - 1,
                       2 * np.random.random((int(wymiar / 2))) - 1])

    error_value_neurons= np.array([np.zeros(len(weights[0])), np.zeros(int(len(weights[0]) / 2)), 0])

    for iteracja in range(numberiteration):
        for iteratordata in range(len(treningData)):
            value_of_network,neuron_responses=predict(treningData[iteratordata],weights)
            #print(value_of_network)
            error_value_neurons[2]= answer[iteratordata]-value_of_network

            for level in range(len(weights)-2,-1,-1):
                error_value_neurons[level]=setErrorValueForLevel(error_value_neurons,level,weights)

            weights=changweigts(treningData[iteratordata],weights,error_value_neurons,neuron_responses)
    return weights

def setErrorValueForLevel(neuron_responses, level, weights):
    if(level==1):
        neuron_responses[level]= neuron_responses[level + 1] * weights[level + 1]
        return neuron_responses[level]
    else:
        for index in range(len(neuron_responses[level])):
            weights_of_neuron=np.zeros(len(neuron_responses[level + 1]))

            for i in range(len(weights[level+1])):
                weights_of_neuron[i]=weights[level+1][i][index]

            neuron_responses[level][index]=sum(neuron_responses[level + 1] * weights_of_neuron)

        return neuron_responses[level]

def changweigts(data, weights, error_value_neuron, neuron_responses):
    for level in range(len(weights)):
        if(level==0):
            for row in range(len(weights[level])):
                weights[level][row]=weights[level][row]+etta*error_value_neuron[level]*derivative_sigmoid(sum(weights[level][row]*data))*data

        elif(level==2):
            weights[level]+= etta * error_value_neuron[level] * derivative_sigmoid(sum(weights[level] * neuron_responses[level - 1])) * neuron_responses[level - 1]

        else:
            for row in range(len(weights[level])):
                weights[level][row]+= etta * error_value_neuron[level-1] * derivative_sigmoid(sum(weights[level][row] * neuron_responses[level - 1])) * neuron_responses[level - 1]

    return weights
