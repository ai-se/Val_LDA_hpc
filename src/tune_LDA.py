from __future__ import print_function, division
from DE import differential_evolution
from random import shuffle
from demos import cmd
from val_lda import Cross_exp
from models import LDA_tune
import pickle
from pdb import set_trace



def exp(set):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    proc_num = 10

    if rank==0:
        file = '/share2/zyu9/Datasets/SE/'+str(set)+'.txt'
        # file = '/Users/zhe/PycharmProjects/Datasets/StackExchange/'+str(set)+'.txt'
        tmp=Cross_exp()
        tmp.load(file)
        num = len(tmp.csr_mat)
        x=range(num)
        sequence=[]
        for i in xrange(10):
            shuffle(x)
            sequence.append(x)
        kw={
            'filepath' : file,
            'sequence' : sequence,
            'term' : 6
        }
        dec,obj = differential_evolution(**kw)
        for i in xrange(proc_num - 1):
            comm.send("finished", dest=i + 1)

        untuned1 = LDA_tune(**kw)
        untuned1.dec=[100,0.1,0.01]
        obj1 = untuned1.getobj()
        untuned3 = LDA_tune(**kw)
        untuned3.dec=[200,0.1,0.01]
        obj3 = untuned1.getobj()

        sequence2=[]
        for i in xrange(10):
            shuffle(x)
            sequence2.append(x)
        kw2= {
            'filepath' : file,
            'sequence' : sequence2,
            'term' : 6
        }
        tuned2 = LDA_tune(**kw2)
        tuned2.dec= dec
        obj2 = tuned2.getobj()

        untuned100 = LDA_tune(**kw2)
        untuned100.dec=[100,0.1,0.01]
        obj100 = untuned100.getobj()
        untuned200 = LDA_tune(**kw2)
        untuned200.dec=[200,0.1,0.01]
        obj200 = untuned200.getobj()

        results={"tuned_train": obj, "tuned_dec": dec, "untuned100_train": obj1, "untuned200_train": obj3, "untuned100_test":obj100, "untuned200_test":obj200, "tuned_test": obj2}

        with open("../dump/tuneLDA_"+str(set)+".pickle", "w") as f:
            pickle.dump(results,f)

    else:
        while True:
            tunee = comm.recv(source=0)
            if type(tunee)==type("str"):
                break
            era = 0
            topics = []
            while True:
                i = era * proc_num + rank
                if i + 1 > len(tunee.sequence):
                    break
                topics.extend(tunee.model.stability_score(tunee.sequence[i]))
                era = era + 1
            comm.send(topics, dest=0)













if __name__ == "__main__":
    eval(cmd())