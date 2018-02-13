from evaluate import f1_community, jc_community, nmi_community
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../data/labels.pkl', 'rb') as f:
	labellist = pickle.load(f)

dim = len(labellist)
newllist = []
for i in xrange(dim):
	labels = labellist[i]
	unique = np.unique(labels)
	newl = [[0]*len(labels) for j in range(len(unique))]
	for j in xrange(len(labels)):
		newl[labels[j]][j] = 1
	newllist.append(newl)

cross_f1 = []
cross_jc = []
cross_nmi = []
for i in xrange(dim):
	cross_f1 += [[0] * dim]
	cross_jc += [[0] * dim]
	cross_nmi += [[0] * dim]
	for j in xrange(dim):
		if i != j:
			cross_f1[i][j] = f1_community(newllist[i], newllist[j])
			cross_jc[i][j] = jc_community(newllist[i], newllist[j])
			cross_nmi[i][j] = nmi_community(newllist[i], newllist[j])
		else:
			cross_f1[i][j] = 1
			cross_jc[i][j] = 1
			cross_nmi[i][j] = 1

with open('../data/cross.txt', 'w') as f:
	for p in xrange(dim):
		for q in xrange(dim):
			f.write(str(cross_f1[p][q])+' ')
		f.write('\n')
	f.write('------------\n')
	for p in xrange(dim):
		for q in xrange(dim):
			f.write(str(cross_jc[p][q])+' ')
		f.write('\n')
	f.write('------------\n')
	for p in xrange(dim):
		for q in xrange(dim):
			f.write(str(cross_nmi[p][q])+' ')
		f.write('\n')

plt.imshow(cross_f1, cmap='Blues', interpolation='nearest')
plt.savefig("../plots/f1_cross.png", bbox_inches='tight')

plt.imshow(cross_jc, cmap='Blues', interpolation='nearest')
plt.savefig("../plots/jc_cross.png", bbox_inches='tight')

plt.imshow(cross_nmi, cmap='Blues', interpolation='nearest')
plt.savefig("../plots/nmi_cross.png", bbox_inches='tight')