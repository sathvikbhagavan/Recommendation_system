import numpy as np

f = open("data.X","r")
f1 = open("mydata.train","w+")
f2 = open("mydata.test","w+")
# f3 = open("mydata.test","w+")
fr = f.readlines()
# ind = np.random.permutation(len(fr))
ind_train = fr[:int(0.7*len(fr))]
# ind_val = ind[int(0.7*len(fr)):int(0.85*len(fr))]
ind_test = fr[int(0.7*len(fr)):]
# ind_train = np.sort(ind_train)
# ind_val = np.sort(ind_val)
# ind_test = np.sort(ind_test)
# print(ind_train)
# print(ind_val)
# print(ind_test)
#X_train = fr[ind_train]
#X_test = fr[ind_test]
X_train = []
# X_val = []
X_test = []
for i1 in range(len(ind_train)):
	X_train.append(fr[ind_train[i1]])
for j1 in range(len(ind_test)):
	X_test.append(fr[ind_test[j1]])
# for k1 in range(len(ind_val)):
# 	X_val.append(fr[ind_val[k1]])
for i in range(len(ind_train)):
	f1.write(X_train[i])
f1.close()
for j in range(len(ind_test)):
	f2.write(X_test[j])
f2.close()
# for k in range(len(ind_val)):
# 	f2.write(X_val[k])
# f2.close()

