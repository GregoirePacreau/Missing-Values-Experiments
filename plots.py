import matplotlib.pyplot as plt
import pickle
import pandas
import os

def show_available():
    filelist = os.listdir("results/")
    available = {}
    for i, filename in enumerate(filelist):
        if filename.endswith(".pkl"):
            diconame = filename.replace('_', '","')
            diconame = diconame.replace('=', '":"')
            diconame = '{"' + diconame + '"}'
            available[i] = eval(diconame)
    return pandas.DataFrame(available)

def plot_dico():
    filename = ""
    with open(filename, rb) as file:
        res_dico = pickle.load(file)

    
