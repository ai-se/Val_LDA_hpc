from __future__ import print_function, division
from pdb import set_trace
import copy
import numpy as np


def calculate(topics=[], lis=[], count1=0):
    count = 0
    for i in topics:
        if i in lis:
            count += 1
    if count >= count1:
        return count
    else:
        return 0


def recursion(topic=[], index=0, count1=0, data=[]):
    count = 0
    # print(data)
    # print(topics)
    d = copy.deepcopy(data)
    d.pop(index)
    for l, m in enumerate(d):
        # print(m)
        for x, y in enumerate(m):
            if calculate(topics=topic, lis=y, count1=count1) != 0:
                count += 1
                break
                # data[index+l+1].pop(x)
    return count




def jaccard(a, score_topics=[], term=0):
    x=term
    l = score_topics
    data = []
    for i in range(0, len(l), int(a)):
        l1 = []
        for j in range(int(a)):
            l1.append(l[i + j])
        data.append(l1)
    j_score = []
    for i, j in enumerate(data):
        for l, m in enumerate(j):
            sum = recursion(topic=m, index=i, count1=x, data=data)
            if sum != 0:
                j_score.append(sum / float(9))
            '''for m,n in enumerate(l):
                if n in j[]'''
    if len(j_score) == 0:
        j_score = [0]
    return np.median(j_score)


