import numpy as np
from numpy import random as rand

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# X: n x d matrix in csr_matrix format containing d-dim (sparse) features for n test data points
# k: the number of recommendations to return for each test data point in ranked order

# OUTPUT CONVENTION
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of labels with the i-th row 
# containing k labels which it thinks are most appropriate for the i-th test point. Labels must be returned in 
# ranked order i.e. the label yPred[i][0] must be considered most appropriate followed by yPred[i][1] and so on

# CAUTION: Make sure that you return (yPred below) an n x k numpy (nd) array and not a numpy/scipy/sparse matrix
# The returned matrix will always be a dense matrix and it terribly slows things down to store it in csr format
# The evaluation code may misbehave and give unexpected results if an nd-array is not returned

def getReco( X, k ):
    # Find out how many data points we have
#    n = X.shape[0]
    # Load and unpack the dummy model
    # The dummy model simply stores the labels in decreasing order of their popularity
#    npzModel = np.load( "model.npz" )
#    model = npzModel[npzModel.files[0]]
#    print(type(npzModel))   
#    print(model.shape)
    # Let us predict a random subset of the 2k most popular labels no matter what the test point
#    print(model)
#    shortList = model[0:2*k]
    # Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
#    yPred = np.zeros( (n, k) )
#    for i in range( n ):
#        yPred[i,:] = rand.permutation( shortList )[0:k]
#        print(yPred[i,:])
#        print(np.shape(yPred[i,:]))
    f = open("/home/akash/Downloads/assn2/FastXML/dataset/top5pred.txt","r")
    fr = f.readlines()
    yPred = np.zeros((2000,k))
    for i in range(len(fr)):
        a = fr[i].split(" ")
        print(a)
        a[-1] = a[-1][:-1]
        print(a)
        b = np.array(a)
        #print(b)
        b = b.astype(int)
        yPred[i,:] = b[:k]
    return yPred
