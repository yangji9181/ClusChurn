import numpy as np
import matplotlib.pyplot as plt
import pickle

cutperc = 0.1
core_friend = set()
core_user = set()
with open('../data/core_user', 'r') as f:
	f.next()
	for line in f:
		core_user.add(line.strip())

with open('../data/core_friend', 'r') as f:
	f.next()
	for line in f:
		core_friend.add(line.strip())

y_user = []
y_friend = []
y_rest = []
x_user = []
x_friend = []
x_rest = []
with open('../data/core_degree', 'r') as f:
	count = 0
	f.next()
	for line in f:
		words = line.split(',')
		if words[0].strip() in core_user:
			y_user.append(int(words[1].strip()))
			x_user.append(count)
		elif words[0].strip() in core_friend:
			y_friend.append(int(words[1].strip()))
			x_friend.append(count)
		else:
			y_rest.append(int(words[1].strip()))
			x_rest.append(count)
		count += 1

cutoff = np.log(count*cutperc)
percent = map(lambda i: i < cutoff, x_friend).count(True) *1.0 / len(x_friend)
print(cutoff, percent)


plt.scatter(np.log(y_friend), np.log(x_friend), marker='d', alpha=0.1, facecolors='none', edgecolors='r', label='K direct friends of new users')
plt.scatter(np.log(y_rest), np.log(x_rest), marker='.', alpha=0.05, facecolors='none', edgecolors='g', label='All other users')
plt.axhline(cutoff, ls='--', c='b', label='Cutoff at top 10% users \nwith highest degrees')
plt.legend(fontsize=15, )
plt.ylabel('Log(Rank By Degree)', fontsize=18)
plt.xlabel('Log(Degree)', fontsize=18)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
#plt.scatter(x_user, y_user, marker='*', facecolors='none', edgecolors='b')
plt.savefig('../plots/fig_core_original.png', format='png', bbox_inches='tight')
plt.show()
plt.clf()