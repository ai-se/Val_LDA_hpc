from __future__ import print_function
from __future__ import absolute_import, division
from random import uniform
from time import time
import numpy as np
from pdb import set_trace
from val_lda import Cross_exp
from stability import jaccard


class Model(object):
    def any(self):
        while True:
            for i in range(0,self.decnum):
                self.dec[i]=uniform(self.bottom[i],self.top[i])
            if self.check(): break
        return self

    def __init__(self):
        self.bottom=[0]
        self.top=[0]
        self.decnum=0
        self.objnum=0
        self.dec=[]
        self.lastdec=[]
        self.obj=[]
        self.any()

    def eval(self):
        return sum(self.getobj())

    def copy(self,other):
        self.dec=other.dec[:]
        self.lastdec=other.lastdec[:]
        self.obj=other.obj[:]
        self.bottom=other.bottom[:]
        self.top=other.top[:]
        self.decnum=other.decnum
        self.objnum=other.objnum

    def getobj(self):
        return []

    def getdec(self):
        return self.dec

    def check(self):
        for i in range(0,self.decnum):
            if self.dec[i]<self.bottom[i] or self.dec[i]>self.top[i]:
                return False
        return True

"Models:"
class LDA_tune(Model):
    def __init__(self,filepath="",sequence=[],term=6):
        self.bottom=[100,0,0]
        self.top=[100,1,1]
        self.decnum=3
        self.objnum=1
        self.dec=[0]*self.decnum
        self.file=filepath
        self.lastdec=[]
        self.obj=[]
        self.sequence=sequence
        self.term=term
        self.any()

    def getobj(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        proc_num = 10

        if self.dec==self.lastdec:
            return self.obj
        self.dec[0]=int(self.dec[0])
        self.model=Cross_exp([1,self.dec[0],1,1,self.dec[1],self.dec[2]])
        self.model.load(self.file)
        self.model.pre_lda()
        for i in xrange(proc_num-1):
            comm.send(self, dest=i+1)
        era = 0
        topics=[]
        while True:
            i = era * proc_num + rank
            if i + 1 > len(self.sequence):
                break
            topics.extend(self.model.stability_score(self.sequence[i]))
            era=era+1
        for i in range(proc_num - 1):
            tmp = comm.recv(source=i + 1)
            topics.extend(tmp)
        self.obj = [jaccard(self.dec[0],topics,self.term)]
        return self.obj

    def getobj_local(self):
        if self.dec==self.lastdec:
            return self.obj
        self.dec[0]=int(self.dec[0])
        self.model=Cross_exp([1,self.dec[0],1,1,self.dec[1],self.dec[2]])
        self.model.load(self.file)
        self.model.pre_lda()
        topics=[]
        for seq in self.sequence:
            topics.extend(self.model.stability_score(seq))

        self.obj = [jaccard(self.dec[0],topics,self.term)]
        return self.obj, topics


