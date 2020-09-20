'''
    Package: cs771 - assn 2
    Module: plotData
    Author: Puru
    Institution: CSE, IIT Kanpur
    License: GNU GPL v3.0
    
    Various utilities for multi-label learning problems
'''

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from scipy import sparse as sps

def load_data( filename, d, L ):
    X, _ = load_svmlight_file( "%s.X" % filename, multilabel = True, n_features = d, offset = 1 )
    y, _ = load_svmlight_file( "%s.y" % filename, multilabel = True, n_features = L, offset = 1 )
    return (X, y)

def dump_data( X, y, filename ):
    (n, d) = X.shape
    (n1, L) = y.shape
    assert n1 == n, "Mismatch in number of feature vectors and number of label vectors"
    dummy = sps.csr_matrix( (n, 1) )
    dump_svmlight_file( X, dummy, "%s.X" % filename, multilabel = True, zero_based = True, comment = "%d, %d" % (n, d) )
    dump_svmlight_file( y, dummy, "%s.y" % filename, multilabel = True, zero_based = True, comment = "%d, %d" % (n, L) )

# Not the best way to do things in Python but I could not find a neater workaround
# Let me know if you know one that avoids a messy loop
def removeDuplicates( pred, imputation ):
    # Create a new array so that the original input array pred is unaffected
    deDup = np.ones( pred.shape ) * imputation
    for i in range( pred.shape[0] ):
        # Retain only the first occurrence of a label in every row
        (u, idx) = np.unique( pred[i,:], return_index = True )
        deDup[i,idx] = u
    return deDup

# Validate that data is nice and well behaved
# Return copies of the gold and predicted labels that removes duplicates
# Original data is not affected i.e. this method can be called repeatedly
# without affecting the arguments sent as inputs
def validateAndCleanup( yGold, yPred, k ):
    (n, L) = yGold.shape

    # Make sure the prediction matrix is in correct  shape
    assert yPred.shape[0] == n, "Mismatch in number of test data points and number of predictions"
    assert yPred.shape[1] >= k, "Less predictions received than were expected"

    # Introduce a dummy label that is never present in any test point
    # Copy data so that the original matrix that was sent in is unaffected
    yGoldNew =  sps.csr_matrix( (yGold.data, yGold.indices, yGold.indptr), shape = (n, L + 1), copy = True )

    # Penalize duplicates in yPred by replacing them with predictions of the dummy label
    yPredNew = removeDuplicates( yPred, L )

    return (yGoldNew, yPredNew)

# For a given value of k, return prec@1, prec@2, ..., prec@k
def getPrecAtK( yGold, yPred, k ):
    n = yGold.shape[0]
    (yGoldNew, yPredNew) = validateAndCleanup( yGold, yPred, k )

    # Use some fancy indexing (yes, this is the formal term for the technique)
    # to find out where all did we correct predict an item liked by the user
    # Python indexing with arrays creates copies of data so we are safe
    wins = yGoldNew[ np.arange( n )[:,np.newaxis], yPredNew ]
    # Find how many times did we correctly predict any item at the blah-th position
    totWins = np.sum( wins, axis = 0 )
    # Find how many times did we correctly predict any item at any one of the top blah positions
    cumWins = np.cumsum( totWins )
    
    # Normalize properly and return
    precAtK = cumWins / (n * (np.arange( k ) + 1))
    return precAtK

# For a given value of k, return mprec@1, mprec@2, ..., mprec@k
def getMPrecAtK( yGold, yPred, k ):
    L = yGold.shape[1]
    (yGoldNew, yPredNew) = validateAndCleanup( yGold, yPred, k )
    mPrecAtK = np.zeros( k )

    # For all real labels (exclude the dummy label)
    for label in range( L ):
        # Find users who like this item
        usersThatLikeThisItem = (yGoldNew[:, label] == 1).toarray().reshape( -1 )
        n_label = np.sum( yGoldNew[:, label] )

        # If there exist users who actually like this item
        if n_label > 0:
            # Find all users for whom we predicted this item
            # Create a new array so that the array yPredNew is unaffected since it has to be reused
            winsThisItem = np.zeros( yPredNew.shape )
            winsThisItem[ yPredNew == label ] = 1
            # Remove cases of users who do not like this item
            winsThisItem[ ~usersThatLikeThisItem, : ] = 0
            # Find how many times did we correctly predict this item at the blah-th position
            totWinsThisItem = np.sum( winsThisItem, axis = 0 )
            # Find how many times did we correctly predict this item at any one of the top blah positions
            cumWins = np.cumsum( totWinsThisItem )
            # Add the wins to mprec@blah
            mPrecAtK += cumWins / n_label

    return mPrecAtK / L

        