import numpy as np

f1 = open("/home/akash/Downloads/assn2/FastXML/dataset/mydata.train.X","r")
f2= open("/home/akash/Downloads/assn2/FastXML/dataset/mydata.train.y","r")
fx = open("/home/akash/Downloads/assn2/FastXML/dataset/data.X","w+")
fy = open("/home/akash/Downloads/assn2/FastXML/dataset/data.y","w+")
fr1 = f1.readlines()
fr2 = f2.readlines()
for i in range(len(fr1)):
    fx.write(fr1[i][1:])
for i in range(len(fr2)):
    fy.write(fr2[i][1:])    
fx.close()
fy.close()
