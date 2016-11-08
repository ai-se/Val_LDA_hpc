from __future__ import print_function, division
import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from random import shuffle
from ABCD import ABCD
from funcs import *
import lda
from demos import cmd



class Cross_exp(object):

    def __init__(self,opt=[0,4000,1,1,0.1,0.01]):
        ## opt: [isLDA, #topics, isL2, isSMOTE]
        self.opt=opt
        self.clf = svm.SVC(kernel='linear')


    def split_cross(self, num, fold, i):
        x = range(num)
        size = int(num / fold)
        less = x[i * size:i * size + size]
        more = list(set(x) - set(less))
        more = np.array(more)
        less = np.array(less)
        return more, less

    def crossval(self, fold=5):
        num = len(self.label)
        tmp = range(num)
        shuffle(tmp)
        label = self.label[tmp]
        data = self.csr_mat[tmp]
        result=[]
        for i in xrange(fold):
            test, train = self.split_cross(num, fold, i)
            if self.opt[3]:
                data_train,label_train = smote_most(data[train],label[train])
                self.clf.fit(data_train, label_train)
            else:
                self.clf.fit(data[train], label[train])
            prediction = self.clf.predict(data[test])
            abcd = ABCD(before=label[test], after=prediction)
            F2 = abcd()['pos'].stats()['F2']
            result.append(F2)
        return result

    def crossval_hpc(self, fold=5):
        num = len(self.label)
        tmp = range(num)
        shuffle(tmp)
        label = self.label[tmp]
        data = self.csr_mat[tmp]

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        proc_num = 5

        result=[]
        era=0
        while True:
            i=era*proc_num+rank
            if i+1 > fold:
                break
            test, train = self.split_cross(num, fold, i)
            if self.opt[3]:
                data_train, label_train = smote_most(data[train], label[train])
                self.clf.fit(data_train, label_train)
            else:
                self.clf.fit(data[train], label[train])
            prediction = self.clf.predict(data[test])
            abcd = ABCD(before=label[test], after=prediction)
            F2 = abcd()['pos'].stats()['F2']
            result.append(F2)
            era+=1
        if rank == 0:
            for i in range(proc_num-1):
                tmp=comm.recv(source=i+1)
                result.extend(tmp)
            return result
        else:
            comm.send(result,dest=0)


    def pre_lda(self):
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False)
        self.csr_mat = tfer.fit_transform(self.csr_mat)
        self.csr_mat = self.csr_mat.astype(np.int32)

    def stability_score(self, sequence):
        model = lda.LDA(n_topics=self.opt[1], n_iter=200, alpha=self.opt[4], eta=self.opt[5])
        topics=[]
        data = self.csr_mat[sequence]
        model.fit(data)
        topic_word = model.topic_word_
        for topic in topic_word:
            topics.append(np.argsort(topic)[::-1][:9])
        return topics



    def load(self,filepath):
        self.label, self.csr_mat = readfile(filepath)
        self.csr_mat=np.array(self.csr_mat)

    def preprocess(self):
        if self.opt[0]:

            tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False)
            self.csr_mat=tfer.fit_transform(self.csr_mat)
            ldaer=lda.LDA(n_topics=self.opt[1], n_iter=200, alpha=self.opt[4], eta=self.opt[5])
            self.csr_mat = self.csr_mat.astype(np.int32)
            self.csr_mat = csr_matrix(ldaer.fit_transform(self.csr_mat))
            if self.opt[2]:
                self.csr_mat=l2normalize(self.csr_mat)
            self.voc = np.array([ ",".join([tfer.vocabulary_.keys()[word] for word in np.argsort(topic)[::-1][:8]]) for topic in ldaer.topic_word_])
        else:
            tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False)
            tfidf = tfidfer.fit_transform(self.csr_mat)
            weight = tfidf.sum(axis=0).tolist()[0]
            kept = np.argsort(weight)[::-1][:self.opt[1]]
            self.voc = np.array(tfidfer.vocabulary_.keys())[np.argsort(tfidfer.vocabulary_.values())][kept]
            ##############################################################

            ### Term frequency as feature, L2 normalization ##########
            tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                            vocabulary=self.voc)
            self.csr_mat=tfer.fit_transform(self.csr_mat)

def exp_mt(method,set="SE1"):
    opt=[0,4000,1,1,0.1,0.01]
    if method=="SVM_SMOTE":
        opt=[0,4000,1,1,0.1,0.01]
    if method=="LDAl2_SMOTE_100":
        opt=[1,100,1,1,0.1,0.01]
    if method=="LDA_SMOTE_100":
        opt=[1,100,0,1,0.1,0.01]
    if method=="LDAl2_SMOTE_200":
        opt=[1,200,1,1,0.1,0.01]
    if method=="LDA_SMOTE_200":
        opt=[1,200,0,1,0.1,0.01]
    A=Cross_exp(opt)
    A.load('/share2/zyu9/Datasets/SE/'+str(set)+'.txt')
    A.preprocess()
    result = []
    for i in xrange(6):
        result.extend(A.crossval_hpc())
    with open("../dump/"+str(set)+"_"+str(method)+".pickle", "w") as f:
        pickle.dump(result,f)

def exp_hpc(method,set="SE1"):
    opt=[0,4000,1,1,0.1,0.01]
    if method=="SVM_SMOTE":
        opt=[0,4000,1,1,0.1,0.01]
    if method=="LDAl2_SMOTE_100":
        opt=[1,100,1,1,0.1,0.01]
    if method=="LDA_SMOTE_100":
        opt=[1,100,0,1,0.1,0.01]
    if method=="LDAl2_SMOTE_200":
        opt=[1,200,1,1,0.1,0.01]
    if method=="LDA_SMOTE_200":
        opt=[1,200,0,1,0.1,0.01]
    A=Cross_exp(opt)
    A.load('/share2/zyu9/Datasets/SE/'+str(set)+'.txt')
    A.preprocess()
    result = []

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    proc_num = 5
    era=0
    while True:
        i=era*proc_num+rank
        if i+1 > 5:
            break
        result.extend(A.crossval())
        era+=1
    if rank == 0:
        for i in range(proc_num-1):
            tmp=comm.recv(source=i+1)
            result.extend(tmp)
        with open("../dump/"+str(set)+"_"+str(method)+".pickle", "w") as f:
            pickle.dump(result,f)
    else:
        comm.send(result,dest=0)

def exp(method,set="SE0"):
    opt=[0,4000,1,1,0.1,0.01]
    if method=="SVM_SMOTE":
        opt=[0,4000,1,1,0.1,0.01]
    if method=="LDAl2_SMOTE":
        opt=[1,7,1,1,0.8724171291071249, 0.9018122438660119]
    if method=="LDA_SMOTE":
        opt=[1,200,0,1,0.1,0.01]
    A=Cross_exp(opt)
    A.load('/Users/zhe/PycharmProjects/Datasets/StackExchange/'+str(set)+'.txt')
    A.preprocess()
    result = []
    result.extend(A.crossval())
    set_trace()





if __name__ == "__main__":
    eval(cmd())
