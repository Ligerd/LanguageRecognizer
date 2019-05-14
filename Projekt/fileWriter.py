import numpy as np
def filewrite(weights,nameFile=None):
    if nameFile is None:
        nameFile="defaultFileWeights"
        np.save(nameFile,weights)
        print("Wagi zostały zapisane do domyśłnego pliku o nazwie: " + nameFile)
    else:
        np.save(nameFile,weights)
        print("Wagi zostały zapisane do pliku o nazwie: "+ nameFile)
