import kmeans
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../data/labels.pkl', 'rb') as f:
	labellist = pickle.load(f)
	centerlist = pickle.load(f)

#use the variable params to store user params replaced by center params
dim = len(labellist)
num_user = len(labellist[0])
num_param = 0
for centers in centerlist:
	num_param += centers.shape[1]
params = np.empty(shape=(num_user, 1, num_param))
for i in xrange(num_user):
	start_dim = 0
	for j in xrange(dim):
		params[i, 0, start_dim:start_dim+centerlist[j].shape[1]] = centerlist[j][labellist[j][i],:]
		start_dim += centerlist[j].shape[1]

#kmeans.Kmeans(input=params, output='../data/multiview.pkl', range_n_clusters=[6])

with open('../data/multiview.pkl', 'rb') as f:
	labels = pickle.load(f)[0]
	centers = pickle.load(f)[0]

#draw churn analysis bar plot
unique = np.unique(labels)
with open('../data/churns.pkl', 'rb') as f:
	churn_label = pickle.load(f)
churn_rate = np.array([0]*len(unique), dtype=np.float32)
churn_count = np.array([0]*len(unique))
for i in xrange(len(labels)):
	churn_count[labels[i]] += 100
	churn_rate[labels[i]] += churn_label[i]
avg_churn_rate = sum(churn_rate) / sum(churn_count)
churn_rate = churn_rate / churn_count
#order labels by churn rate
churn_ind = np.argsort(churn_rate)
churn_rate = churn_rate[churn_ind]
labels = np.array(map(lambda i: list(churn_ind).index(i), labels))
centers = centers[churn_ind]
plt.clf()
x = range(len(churn_rate))
above = np.maximum(churn_rate - avg_churn_rate, 0)
below = np.minimum(churn_rate, avg_churn_rate)
plt.bar(x, below, color='g', alpha=0.8)
plt.bar(x, above, color='r', alpha=0.8, bottom=below)
plt.plot([-0.5, 5.5], [avg_churn_rate, avg_churn_rate], 'k--', label='Average Churn Rate')
plt.xlim((-0.5, 5.5))
plt.xticks(x, ('All-star', 'Chatter', 'Bumper', 'Sleeper', 'Swiper', 'Invitee'), fontsize=20, rotation=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('../plots/fig_churn.png')

with open('../data/multiview.pkl', 'wb') as f:
	pickle.dump([labels], f)
	pickle.dump([centers], f)

print(list(labels).count(0), list(labels).count(1), list(labels).count(2), list(labels).count(3), list(labels).count(4), list(labels).count(5))


#draw pie chart
portion = [0]*len(unique)
for i in unique:
	portion[i] = np.count_nonzero(labels == i)
plt.clf()
plt.pie(portion, labels = unique)
plt.savefig('../plots/multiview_portion.png')
print(np.array(portion, dtype=np.float32)/sum(portion))

#mark the typical clusters
labellist = []
for i in xrange(centers.shape[0]):
	center = centers[i,:].squeeze()
	labels = []
	start_dim = 0
	for j in xrange(dim):
		cc = center[start_dim:start_dim+centerlist[j].shape[1]]
		start_dim += centerlist[j].shape[1]
		id_nearest = -1
		min_dis = 100
		for p in xrange(centerlist[j].shape[0]):
			if np.linalg.norm(cc-centerlist[j][p,:]) < min_dis:
				min_dis = np.linalg.norm(cc-centerlist[j][p,:])
				id_nearest = p
		labels.append(id_nearest)
	labellist.append(labels)

with open('../data/multiview.txt', 'w') as f:
	for labels in labellist:
		for label in labels:
			f.write(str(label)+' ')
		f.write('\n')

#with open('../data/params.pkl', 'rb') as f:
#	params = pickle.load(f)
softlabels = np.empty(shape=(params.shape[0], centers.shape[0]), dtype=np.float32)
for i in xrange(params.shape[0]):
	userfeat = params[i,:,:].reshape(params.shape[1]*params.shape[2])
	norm = 0
	values = np.empty(shape=(centers.shape[0]), dtype=np.float32)
	for j in xrange(centers.shape[0]):
		values[j] = np.power((1+np.sum(np.power(centers[j,:]-userfeat, 2))),-1)
		norm += values[j]
	softlabels[i,:] = values / norm

with open('../data/soft.pkl', 'wb') as f:
	pickle.dump(softlabels, f)
labels = map(lambda i: list(i).index(max(i)), list(softlabels))

print(list(labels).count(0), list(labels).count(1), list(labels).count(2), list(labels).count(3), list(labels).count(4), list(labels).count(5))




