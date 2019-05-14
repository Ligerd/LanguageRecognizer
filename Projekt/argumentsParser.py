import argparse
from predict import predict
from fileParser import count_letters
from fileParser import readData
from fileWriter import filewrite
from training import trainingWeights
import os
import numpy as np
def argument_analysis():
    parser=argparse.ArgumentParser(description='Program dla rozpoznowania języka za pomocą sieci neuronowej')
    parser.add_argument('-t','--traning',action='store_true', help='Argument dla uruchomienia programu w trybie trenowania.')
    parser.add_argument('-p', '--prediction', action='store_true', help='Argument dla uruchomienia programu w trybie rozpoznawania czy to jest język Angielski')
    parser.add_argument('-f','--files',nargs='+', help='Pliki wejściowe do programu')
    parser.add_argument('-fw', '--fileweights', nargs=1, help='Plik zawierający wagi sieci neuronowej')
    parser.add_argument('-a','--answers',nargs='+',help='Wymagany odpowiedzi sieci neuronowej')
    parser.add_argument('-n','--iteration',nargs=1,help='Liczba iteraji')
    #n jako liczba iteracji
    parser.add_argument('-w','--writetofile',nargs=1,help='Zapisanie wag do wybranego pliku')
    res=parser.parse_args()
    if res.traning:
        if res.files is None or res.answers is None or res.iteration is None:
            print("Parametr -t wymaga dodatkowych parametrów: -f, -a, -i")
            print("Wywołaj program z argumentem -h albo --help dla uzystania informacji o argumentachach ;)")
            exit(0)
        for file in res.files:
            if not os.path.isfile(file):
                print("Plik nie jest plikiem o nazwie: " + file)
                exit(0)
        for answer in res.answers:
            if not answer.isdigit():
                print("Odpowiedź sieci może przyjmować wartości albo 0 albo 1. Nie poprwana odpowiedź: "+ answer)
                exit(0)
            if int(answer)!=0 and int(answer)!=1:
                print("Odpowiedź sieci może być albo 0 albo 1 dla pliku. Nie poprawna odpowiedź: " + answer)
                exit(0)
        if len(res.files)!=len(res.answers):
            print("Ilość wyjściowych plików dla uczenia sieci muszi być tyle ile odpowiedzi sieci")
            exit(0)
        try:
            value=int(res.iteration[0])
            if value<0:
                print("Ilość iteracji nie może być mniej niż 0")
                exit(0)
        except ValueError:
            print("Ilość iteracji muszi być zadana liczba")
            exit(0)
        filesstatistic=[]
        for file in res.files:
            filesstatistic.append(count_letters(file))
        filesstatistic=np.array(filesstatistic)
        answers=[]
        for answer in res.answers:
            answers.append(int(answer))
        answers=np.array(answers)

        if res.fileweights is None:
            weights=trainingWeights(filesstatistic,answers,int(res.iteration[0]))
            if res.writetofile is None:
                filewrite(weights)
            else:
                filewrite(weights,res.fileweights)
        else:
            if not os.path.isfile(res.fileweights[0]):
                print("Podany plik wejściowy zawierający wagi sieci jest nie poprawny. Nazwa pliku: " + res.fileweights[0])
                exit(0)
            else:
                weights=readData(res.fileweights[0])
                trainingWeights(filesstatistic,answers,int(res.iteration[0]),weights)
                if res.writetofile is None:
                    filewrite(weights)
                else:
                    filewrite(weights, res.fileweights)

    elif res.prediction:
            if res.files is None or res.fileweights is None:
                print("Parametr -p wymaga dodatkowych parametrów -f i -fw")
                print("Wywołaj program z argumentem -h albo --help dla uzystania informacji o argumentachach ;)")
                exit(0)
            if os.path.isfile(res.files[0]) and os.path.isfile(res.fileweights[0]):
                input=count_letters(res.files[0])
                weights=readData(res.fileweights[0])
                w,w2=predict(input,weights)
                print("Prawdopodobieństwo tego że to jest język angielski wynosi:", w)
            else:
                print("Pliki weściowe nie poprawne")
                exit(0)
    else:
        print("Nie wybran rzaden tryb")
        exit(0)