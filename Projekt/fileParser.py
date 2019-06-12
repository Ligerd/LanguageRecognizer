import numpy as np
import collections
import string

def count_letters(filename, case_sensitive=False):
    with open(filename, 'r') as f:
        text = f.read()

    if case_sensitive:
        alphabet = string.ascii_letters
    else:
        alphabet = string.ascii_lowercase
        text = text.lower()

    letter_count = collections.Counter()

    for char in text:
        if char in alphabet:
            letter_count[char] += 1
    licznik = np.zeros((len(alphabet)))
    i=0

    for letter in alphabet:
        if sum(letter_count.values())==0:
            print("Podany plik o nazwie: "+filename +" nie zawiera liter ze znanego dla programy alfabetu.")
            exit(0)
        licznik[i]= letter_count[letter]*100/sum(letter_count.values())
        i+=1
    return licznik

def readData(nameFile):
    data=np.load(nameFile)
    return data