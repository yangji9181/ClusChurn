#evaluate link smoothing intuitions
#compute the portion of linked same-type users

import pickle
import numpy as np

with open('../data/act_data.pkl', 'rb') as f:
	useract = pickle.load(f)
	userkey = pickle.load(f)

linklist = []
with open('../data/link/linkset_sampled', 'r') as f:
	for line in f:
  		words = line.split(',')
  		u1 = words[0].strip()
  		u2 = words[1].strip()
		if u1 in userkey and u2 in userkey:
			linklist.append((userkey.index(u1), userkey.index(u2)))

with open('../data/multiview.pkl', 'rb') as f:
	labels = pickle.load(f)[0]

counts = [0]*len(labels)
for tp in linklist:
	if labels[tp[0]] == labels[tp[1]]:
		counts[labels[tp[0]]] += 1

print(np.array(counts, dtype=np.float)/len(tp))


