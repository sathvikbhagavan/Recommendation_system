import utils
import predict
import time as tm
import numpy as np

# This file is intended to demonstrate how we would evaluate your code
# The data loader needs to know how many feature dimensions and labels are there
d = 16385
L = 3400
(X, y) = utils.load_data( "data", d, L )

# Get recommendations from predict.py and time the thing
tic = tm.perf_counter()
yPred = predict.getReco( X, 5 )
toc = tm.perf_counter()

print( "Total time taken is %.6f seconds " % (toc - tic) )

# Need to do a bit reshaping since what we get is technically a sparse matrix
preck = np.asarray( utils.getPrecAtK( y, yPred, 5 ) ).reshape(-1)
# The macro precision code takes a bit longer to execute due to the for loop over labels
mpreck = np.asarray( utils.getMPrecAtK( y, yPred, 5 ) ).reshape(-1)

# Warning: although it is always true that mprek@i > mprec@j if i > j
# the same is not true for prec@k -- prec@k may go up or down as k increases
# See the definition of mprec@k and prec@k carefully to convince yourself why this is so

print( "prec@1: %0.3f" % preck[0], "prec@3: %0.3f" % preck[2], "prec@5: %0.3f" % preck[4] )
# Dont be surprised if mprec is very small -- it is really hard to do well on rare labels
print( "mprec@1: %0.3e" % mpreck[0], "mprec@3: %0.3e" % mpreck[2], "mprec@5: %0.3e" % mpreck[4] )