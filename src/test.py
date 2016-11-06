

def crossval_hpc(fold):
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
        result.append(i)
        era+=1
    if rank == 0:
        for i in range(proc_num-1):
            tmp=comm.recv(source=i+1)
            result.extend(tmp)
        return result
    else:
        comm.send(result,dest=0)
        comm.Free()


def test_LDA():
    import lda
    from pdb import set_trace
    from random import shuffle
    import numpy as np
    from stability import jaccard
    X = lda.datasets.load_reuters()
    topics=[]
    for i in xrange(10):
        x = range(len(X))
        shuffle(x)
        X = np.array(X)[x]
        model = lda.LDA(n_topics=10, n_iter=100, random_state=1)
        model.fit(X)
        topic_word = model.topic_word_
        for topic in topic_word:
            topics.append(np.argsort(topic)[::-1][:9])

    set_trace()
    x = jaccard(10,topics,6)
    set_trace()

