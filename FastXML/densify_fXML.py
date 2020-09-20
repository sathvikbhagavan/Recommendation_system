import numpy as np

f = open("/home/akash/Downloads/assn2/FastXML/dataset/score_mat.txt","r")
fw = open("/home/akash/Downloads/assn2/FastXML/dataset/top5pred.txt","w+")
fr = f.readlines()
print(fr)
for i in range(len(fr)):
	if i==0:
		continue
	a = fr[i].split(" ")
	tmp1 = []
	tmp2 = np.zeros(len(a))
	for j in range(len(a)):
		b = np.array(a[j].split(":"))
		tmp1.append(b[0])
		tmp2[j] = b[1].astype(float)
	label_ind = np.argsort(-tmp2)[:5]
	print(b)
	print(tmp2)
	print(label_ind)

	tmp = ""
	for j in range(len(label_ind)):
		if j==0:
			tmp=str(tmp1[label_ind[j]])
		else:
			tmp = tmp + " " + str(tmp1[label_ind[j]])
		
	print(tmp)
	tmp = tmp + "\n"
	fw.write(tmp)
fw.close()
