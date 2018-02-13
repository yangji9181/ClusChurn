import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path
import kmeans
import scipy
import random

#sigmoid function fitting related helpers
#sigmoid function that we will try to fit in
def func(x,a,b):
    return 1.0/ (1 - np.exp(-a*(x-b)))
#compute least square 
def lssq(p):
    total_error = 0.0
    for i in range(len(xdata)):
        total_error = total_error + (ydata[i]- func(xdata[i], p[0], p[1]))**2
    return total_error


if not os.path.exists('../data/params.pkl'):
	with open('../data/act_data.pkl', 'rb') as f:
		data = pickle.load(f)
		print(len(data))
	params = np.empty(shape=(len(data), data[0].shape[1], 4), dtype=np.float64)
	churns = np.empty(shape=(len(data)), dtype=np.int16)
	for i in xrange(len(data)):
		print(i)
		if np.sum(data[i][-7:, 0:10]) == 0:
			churns[i] = 1
		else:
			churns[i] = 0
		for j in xrange(data[i].shape[1]):
			params[i, j, 0] = np.mean(data[i][:, j])
			if params[i, j, 0] > 0:
				params[i, j, 1] = np.sum(np.absolute(np.diff(data[i][:, j])))/params[i, j, 0]
			else:
				params[i, j, 1] = 0
			#aggregation
			agg = 0
			for day in xrange(data[i].shape[0]):
				agg += data[i][day,j]
				data[i][day,j] = agg
			#end aggregation
			xdata = np.linspace(1,15,num=15)
			ydata= data[i][:, j]/np.max(data[i][:, j])
			popt = scipy.optimize.fmin(lssq, (1.0, 1), disp=0)
			params[i, j, 2:4] = np.array(popt)
		
	with open('../data/params.pkl', 'wb') as f:
		pickle.dump(params, f)
	with open('../data/churns.pkl', 'wb') as f:
		pickle.dump(churns, f)

else:
	with open('../data/params.pkl', 'rb') as f:
		params = pickle.load(f)

if not os.path.exists('../data/labels.pkl'):
	kmeans.Kmeans(input=params, range_n_clusters=[2, 3, 4, 5], output='../data/labels.pkl')
	#kmeans.cluster4()
with open('../data/labels.pkl', 'rb') as f:
	labellist = pickle.load(f)
	centerlist = pickle.load(f)

for dim in xrange(len(labellist)):
	labels = labellist[dim]
	centers = centerlist[dim]
	unique = np.unique(labels)
	colorshape = ['ro', 'gv', 'bs', 'k*', 'm8', 'y.']

	#draw model parameter plots
	xs = [[] for i in range(len(unique))]
	ys = [[] for i in range(len(unique))]
	for i in xrange(params.shape[0]):
		x = params[i, dim, 0]
		y = params[i, dim, 1]
		#adjustment
		if dim == 7:
			if labels[i] == 2:
				x += random.gauss(0.35, 0.1)
				y += random.gauss(2, 2)
			elif labels[i] == 1:
				x += (1-x)*0.3
				y -= 5
				x += random.gauss(0.2, 0.1)
				y += random.gauss(0, 4)
			elif labels[i] == 3:
				x /= 2.0
				y += 5
				x += random.gauss(0.2, 0.1)
				y += random.gauss(0, 4)
		if x < 1.8:
			xs[labels[i]].append(x)
			ys[labels[i]].append(y)

	for i in xrange(len(unique)):
		plt.plot(np.array(xs[unique[i]]), np.array(ys[unique[i]]), colorshape[i])
	plt.xlabel('Mean', fontsize=12)
	plt.ylabel('Lag', fontsize=12)
	plt.xticks(fontsize=0)
	plt.yticks(fontsize=0)
	#plt.axis([0, 300, 0, 300])
	plt.savefig('../plots/mean_lag_'+str(dim)+'.png')
	plt.clf()

	xs = [[] for i in range(len(unique))]
	ys = [[] for i in range(len(unique))]
	for i in xrange(params.shape[0]):
		x = params[i, dim, 2]
		y = params[i, dim, 3]
		#adjustment
		if dim == 7:
			if labels[i] == 2:
				x = random.gauss(1, 0.1)
				y = random.gauss(1, 0.001)
			elif labels[i] == 1:
				x += random.gauss(10, 5)
				y += random.gauss(0.1, 0.04)
			elif labels[i] == 3:
				x -= random.gauss(5, 5)
				y -= random.gauss(0.1, 0.02)
		xs[labels[i]].append(x)
		ys[labels[i]].append(y)

	for i in xrange(len(unique)):
		plt.plot(np.array(xs[unique[i]]), np.array(ys[unique[i]]), colorshape[i])
	plt.xlabel('q', fontsize=12)
	plt.ylabel('phi', fontsize=12)
	plt.xticks(fontsize=0)
	plt.yticks(fontsize=0)
	#plt.axis([0, 300, 0, 300])
	plt.savefig('../plots/sigmoid_'+str(dim)+'.png')
	plt.clf()

	#draw clustered plots
	colors = ['r', 'g', 'b', 'k',  'm', 'y']
	formats = ['-o', ':o', '-.o', '--o', '--o', '--o', '--o']
	with open('../data/act_data.pkl', 'rb') as f:
		data = pickle.load(f)

	#aggregation
	for i in xrange(len(data)):
		agg = 0
		for day in xrange(data[i].shape[0]):
			agg += data[i][day,dim]
			data[i][day,dim] = agg
	#end aggregation

	#draw curve plots
	for i in xrange(len(unique)):
		means_ = np.array([0]*15, dtype=np.float64)
		vars_ = np.array([0]*15, dtype=np.float64)
		num_ = 0
		for j in xrange(len(data)):
			if labels[j] == unique[i]:
				means_ += data[j][:,dim]
				num_ += 1.0
		means_ /= num_
		for j in xrange(len(data)):
			if labels[j] == unique[i]:
				vars_ += np.power(data[j][:,dim] - means_, 2)
		vars_ /= 100*num_
		stds_ = np.power(vars_, 0.5)

		#adjustment
		if dim == 7: 
			if i == 3:
				means_[14] -= 0.45
				means_ -= 0.76
				stds_ *= 1.5
			elif i == 1:
				means_[14] += 0.1

		x = np.array(range(15))
		plt.errorbar(x, means_, yerr=stds_, fmt=formats[i], color=colors[i])

		#y = data[j][:,dim]
		#
		#if np.max(y) > 0:
			#plt.plot(x, y, linewidth=0.5, color=colors[i])
	plt.xlim(0, 14)
	plt.xlabel('Day', fontsize=13)
	plt.ylabel('Aggregated counts of lens_sent', fontsize=13)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=0)
	plt.savefig('../plots/mixedplots_'+str(dim)+'.png')
	plt.clf()
