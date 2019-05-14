from maths import sigmoid
import numpy as np
def predict(input,weights):
    neuron_responses=np.array([np.zeros(len(weights[0])), np.zeros(int(len(weights[0])/2)),0])
    for level in range(len(weights)):

        if (level==1):
            neuron_responses[level]=find_values(input,weights[level],neuron_responses[level],level)

        elif (level==2):
            wynik=find_values(neuron_responses[level-1],weights[level],neuron_responses[level],level)
            neuron_responses[2]=wynik
            return wynik,neuron_responses
        else:
            neuron_responses[level]=find_values(neuron_responses[level-1],weights[level],neuron_responses[level],level)


def find_values(data,weights,output,level):

    if (level==2):
        wynik=sigmoid(sum(weights * data))
        return wynik

    for i in range(len(weights)):
        output[i]=sigmoid(sum(weights[i] * data))

    return output