from __future__ import print_function, division
from collections import Counter
from pdb import set_trace
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.feature_extraction import FeatureHasher
from sklearn import naive_bayes
from sklearn import tree
from time import time
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import *
from random import randint,uniform,random
import lda


from ABCD import ABCD

# __author__ = 'Zhe Yu'







"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat

"Concatenate two csr into one (equal num of columns)"
def csr_vstack(a,b):
    data=np.array(list(a.data)+list(b.data))
    ind=np.array(list(a.indices)+list(b.indices))
    indp=list(a.indptr)+list(b.indptr+a.indptr[-1])[1:]
    return csr_matrix((data,ind,indp),shape=(a.shape[0]+b.shape[0],a.shape[1]))

"smote only oversample"
def smote_most(data,label,k=5):
    labelCont=Counter(label)
    # num=int(sum(labelCont.values())/len(labelCont.values()))
    num=int(np.max(labelCont.values()))
    labelmade=[]
    balanced=[]
    for l in labelCont:
        id=[i for i,x in enumerate(label) if x==l]
        sub=data[id]
        labelmade+=[l]*num
        if labelCont[l]<num:
            num_s=num-labelCont[l]
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(sub)
            distances, indices = nbrs.kneighbors(sub)
            row=[]
            column=[]
            new=[]
            for i in xrange(num_s):
                mid=randint(0,sub.shape[0]-1)
                nn=indices[mid,randint(1,k)]
                indx=list(set(list(sub[mid].indices)+list(sub[nn].indices)))
                datamade=[]
                for j in indx:
                    gap=random()
                    datamade.append((sub[nn,j]-sub[mid,j])*gap+sub[mid,j])
                row.extend([i]*len(indx))
                column.extend(indx)
                new.extend(datamade)
            mat=csr_matrix((new, (row, column)), shape=(num_s, sub.shape[1]))
            if balanced == []:
                balanced=mat
            else:
                balanced=csr_vstack(balanced,mat)
            balanced=csr_vstack(balanced,sub)
        else:
            ind=np.random.choice(labelCont[l],num,replace=False)
            if balanced == []:
                balanced=sub[ind]
            else:
                balanced=csr_vstack(balanced,sub[ind])
    labelmade=np.array(labelmade)
    return balanced, labelmade

"smote"
def smote(data,num,k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(data)
    distances, indices = nbrs.kneighbors(data)
    row=[]
    column=[]
    new=[]
    for i in xrange(num):
        mid=randint(0,data.shape[0]-1)
        nn=indices[mid,randint(1,k)]
        indx=list(set(list(data[mid].indices)+list(data[nn].indices)))
        datamade=[]
        for j in indx:
            gap=random()
            datamade.append((data[nn,j]-data[mid,j])*gap+data[mid,j])
        row.extend([i]*len(indx))
        column.extend(indx)
        new.extend(datamade)
    mat=csr_matrix((new, (row, column)), shape=(num, data.shape[1]))
    mat.eliminate_zeros()
    return mat



"Load data from file to list of lists"
def readfile(filename='',thres=[0.02,0.05]):
    dict=[]
    label=[]
    targetlabel=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            try:
                row=doc.lower().split(' >>> ')[0]
                label.append(doc.lower().split(' >>> ')[1].split()[0])
                dict.append(row)
            except:
                pass
    labellst=Counter(label)
    n=sum(labellst.values())
    while True:
        for l in labellst:
            if labellst[l]>n*thres[0] and labellst[l]<n*thres[1]:
                targetlabel=l
                break
        if targetlabel:
            break
        thres[1]=2*thres[1]
        thres[0]=0.5*thres[0]

    for i,l in enumerate(label):
        if l == targetlabel:
            label[i]='pos'
        else:
            label[i]='neg'
    label=np.array(label)

    return label, dict

"Load data, multi-label"
def readfile_multilabel(filename='',pre='stem'):
    dict=[]
    label=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            try:
                row=doc.lower().split(' >>> ')[0]
                label.append(doc.lower().split(' >>> ')[1].split())
                if pre=='stem':
                    dict.append(Counter(process(row).split()))
                elif pre=="bigram":
                    tm=process(row).split()
                    temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    dict.append(Counter(temp+tm))
                elif pre=="trigram":
                    tm=process(row).split()
                    #temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    temp2=[tm[i]+' '+tm[i+1]+' '+tm[i+2] for i in xrange(len(tm)-2)]
                    dict.append(Counter(temp2+tm))
                else:
                    dict.append(Counter(row.split()))
            except:
                pass
    label=np.array(label)
    return label, dict




