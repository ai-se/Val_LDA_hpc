from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from demos import cmd
import pickle
from pdb import set_trace



def draw():
    methods = ["SVM_SMOTE","LDA_SMOTE_100","LDAl2_SMOTE_100","LDA_SMOTE_200","LDAl2_SMOTE_200"]
    sets = ['academia', 'apple', 'anime', 'android', 'scifi', 'SE0', 'SE1', 'SE2', 'SE3', 'SE4']
    result={}
    median={}
    iqr={}
    for method in methods:
        result[method]= {}
        median[method] = {}
        iqr[method] = {}

        for set in sets:
            with open('../dump/'+set+'_'+method+'.pickle','r') as f:
                result[method][set]=pickle.load(f)
            median[method][set] = np.median(result[method][set])
            iqr[method][set] = np.percentile(result[method][set],75)-np.percentile(result[method][set],25)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)
    plt.figure()

    x=range(len(sets))
    for method in methods:
        line, = plt.plot(x, median[method].values(), label=method)
        plt.plot(x, iqr[method].values(), "-.", color=line.get_color())

    xtick = median[median.keys()[0]].keys()
    plt.xticks(x, xtick, rotation=70)
    plt.ylabel("F2 score")
    plt.xlabel("Datasets")
    plt.legend(bbox_to_anchor=(0.65, 1), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/LDA2.eps")
    plt.savefig("../figure/LDA2.png")


def draw_LDA():
    sets = ['academia', 'apple', 'anime', 'android', 'scifi', 'SE0', 'SE1', 'SE2', 'SE3', 'SE4']
    result={}

    for set in sets:
        with open('../dump/tuneLDA_'+str(set)+".pickle",'r') as f:
            result[set]=pickle.load(f)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)


    cases=["train","test"]
    methods=["tuned","untuned100","untuned200"]

    for i,case in enumerate(cases):
        plt.figure(i)
        x=range(len(sets))
        for method in methods:
            plt.plot(x, [result[key][method+"_"+case] for key in result], label=method)

        xtick = result.keys()
        plt.xticks(x, xtick, rotation=70)
        plt.ylabel("F2 score")
        plt.xlabel("Datasets")
        plt.legend(bbox_to_anchor=(0.65, 1), loc=1, ncol=1, borderaxespad=0.)
        plt.savefig("../figure/LDA_tune_"+case+".eps")
        plt.savefig("../figure/LDA_tune_"+case+".eps")

if __name__ == "__main__":
    eval(cmd())

