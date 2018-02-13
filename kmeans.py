from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pickle
import sys
import math


class Kmeans():
	def __init__(self, k=1, input=[], output='../data/labels.pkl', range_n_clusters=[2,3,4]):
		labels = []
		centers = []

		params = input
		for dim in xrange(params.shape[1]):
			params_ = params[:, dim, :].squeeze()

			for i in xrange(params_.shape[1]):
				params_[:,i] -= np.min(params_[:,i])
				params_[:,i] /= np.max(params_[:,i])

			if k == 1:
				ss = []
				maxs = 0
				maxn = 2
				for n_clusters in range_n_clusters:
					clusterer = KMeans(n_clusters=n_clusters, random_state=10)
					cluster_labels = clusterer.fit_predict(params_)

					silhouette_avg = silhouette_score(params_, cluster_labels)
					print("For n_clusters ="+str(n_clusters)+", the average silhouette_score is :"+str(silhouette_avg))
					ss.append(silhouette_avg)
					if silhouette_avg > maxs:
						maxs = silhouette_avg
						maxn = n_clusters
				print("Picked K for dim "+str(dim)+" is "+str(maxn))
				k_ = maxn
			else:
				k_ = k

			kmeans = KMeans(n_clusters=k_).fit(params_)

			labels.append(kmeans.labels_)
			centers.append(kmeans.cluster_centers_)

		with open(output, 'wb') as f:
			pickle.dump(labels, f)
			pickle.dump(centers, f)


class cluster4():
	def __init__(self, k=4):
		with open('../data/params.pkl', 'rb') as f:
			params = pickle.load(f)

		labels = np.zeros(shape=(params.shape[0]), dtype=np.int16)
		if len(params.shape) > 2:
			params = params[:, 0, :].squeeze()
		for i in xrange(params.shape[1]):
			params[:,i] -= np.min(params[:,i])
			params[:,i] /= np.max(params[:,i])
			for j in xrange(params.shape[0]):
				if params[j,i]>0.5:
					labels[j] += math.pow(2, i)
		with open('../data/labels.pkl', 'wb') as f:
			pickle.dump(labels, f)