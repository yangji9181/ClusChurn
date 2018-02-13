import pickle
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Baseline(object):
	def __init__(self, params):
		self.params = params
		with open(params['data_x'], 'rb') as f:
			self.x = np.array(pickle.load(f))[:, params['access_day'][0]:params['access_day'][1], params['access_feat'][0]:params['access_feat'][1]]
		with open(params['data_y'], 'rb') as f:
			self.y = np.array(pickle.load(f))
		self.x = self.x.reshape(-1, (params['access_day'][1]-params['access_day'][0])*(params['access_feat'][1]-params['access_feat'][0]))
		self.n = self.x.shape[0]
		seed = 1111
		split_train = int(np.floor(self.params['split_ratio']*self.n))
		ind = np.arange(self.n)
		np.random.seed(seed)
		np.random.shuffle(ind)
		self.ind_train, self.ind_test = ind[:split_train], ind[split_train:]

	def logistic(self):
		model = sklearn.linear_model.LogisticRegression()
		model.fit(self.x[self.ind_train,:], self.y[self.ind_train])
		pred = model.predict(self.x[self.ind_test,:])
		true = self.y[self.ind_test]
		acc = (pred==true).sum()*1.0/len(pred)
		tp = map(lambda i: pred[i] == 1 and true[i] == 1, range(len(pred))).count(True)
		fp = map(lambda i: pred[i] == 1 and true[i] == 0, range(len(pred))).count(True)
		fn = map(lambda i: pred[i] == 0 and true[i] == 1, range(len(pred))).count(True)
		prec = tp*1.0/(tp+fp)
		rec = tp*1.0/(tp+fn)
		print(acc, prec, rec)

	def forest(self):
		model = RandomForestClassifier(max_depth=4, random_state=0)
		model.fit(self.x[self.ind_train,:], self.y[self.ind_train])
		pred = model.predict(self.x[self.ind_test,:])
		true = self.y[self.ind_test]
		acc = (pred==true).sum()*1.0/len(pred)
		tp = map(lambda i: pred[i] == 1 and true[i] == 1, range(len(pred))).count(True)
		fp = map(lambda i: pred[i] == 1 and true[i] == 0, range(len(pred))).count(True)
		fn = map(lambda i: pred[i] == 0 and true[i] == 1, range(len(pred))).count(True)
		prec = tp*1.0/(tp+fp)
		rec = tp*1.0/(tp+fn)
		print(acc, prec, rec)

if __name__ == '__main__':
	params = {}
	params['data_x'] = '../data/act_data.pkl'
	params['data_y'] = '../data/churns.pkl'
	params['access_day'] = (0, 7)
	params['access_feat'] = (0, 12)
	params['split_ratio'] =0.8

	predictor = Baseline(params)
	predictor.forest()

