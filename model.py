import pickle
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

class Dataset(object):
	def __init__(self, params):
		self.load_data(params)

	def load_data(self, params):
		with open(params['data_x'], 'rb') as f:
			self.x = torch.FloatTensor(np.array(pickle.load(f))[:, params['access_day'][0]:params['access_day'][1], params['access_feat'][0]:params['access_feat'][1]])
		with open(params['data_y'], 'rb') as f:
			self.y = torch.FloatTensor(pickle.load(f))
		with open(params['data_z'], 'rb') as f:
			self.z = torch.FloatTensor(pickle.load(f).astype(float))

class LstmNet(object):
	def __init__(self, params):
		self.params = params
		self.build_model()

	def build_model(self):
		self.lstms = torch.nn.ModuleList([torch.nn.LSTM(
				input_size=self.params['emb_size_front'],
				#input_size=self.params['n_feat'],  
				hidden_size=self.params['emb_size'], 
				num_layers=self.params['n_layer'], 
				dropout = self.params['dropout'],
				batch_first=True) 
			for i in range(self.params['n_type'])])
		self.front_linear = torch.nn.Sequential(torch.nn.Linear(in_features=self.params['n_feat'], out_features=params['emb_size_front']), torch.nn.Dropout(self.params['dropout']))
		self.linears = torch.nn.ModuleList([torch.nn.Linear(in_features=self.params['emb_size'], out_features=1) for i in range(self.params['n_type'])])
		self.main_linear = torch.nn.Linear(in_features=self.params['emb_size'], out_features=1)
 		self.optimizer = torch.optim.Adam([
 				{'params': self.lstms.parameters()},
 				{'params': self.linears.parameters()},
 				{'params': self.main_linear.parameters()},
 				{'params': self.front_linear.parameters()}
 			], lr=self.params['learning_rate'])

 	def forward(self, x):
 		emb_front = self.front_linear(x)
 		emb_branch = map(lambda t: t(emb_front)[0][:, self.params['len_seq']-1, :], self.lstms)	#n_type x n_user x n_emb
		#emb_branch = map(lambda t: t(x)[0][:, self.params['len_seq']-1, :], self.lstms)	#for taking off front embedding
		#emb_main = torch.stack(emb_branch).mean(dim = 0)		#for mean pooling
		emb_main = torch.stack(emb_branch).max(dim = 0)[0]	#n_user x n_emb
 		y_pred = map(lambda i: torch.nn.Sigmoid()(self.linears[i](emb_branch[i])), range(self.params['n_type']))
		z_pred = torch.nn.Sigmoid()(self.main_linear(emb_main))
		return y_pred, z_pred

 	def get_loss(self, x, y, z):
 		y_pred, z_pred = self.forward(x)
 		loss_branch = map(lambda i: torch.nn.MSELoss()(y_pred[i], y[:,i]), range(self.params['n_type']))
 		loss_main = torch.nn.MSELoss()(z_pred, z)
 		return (loss_main + self.params['lambda']*sum(loss_branch))
 		
class Predictor(object):
	def __init__(self, params):
		self.params = params

	def build_input(self):
		data = Dataset(self.params)
		self.params['n_user'] = data.x.size()[0]
		self.params['len_seq'] = data.x.size()[1]
		self.params['n_feat'] = data.x.size()[2]
		seed = 1111
		split_train = int(np.floor(self.params['split_ratio']*self.params['n_user']))
		ind = np.arange(self.params['n_user'])
		np.random.seed(seed)
		np.random.shuffle(ind)
		self.ind_train, self.ind_test = ind[:split_train], ind[split_train:]

		self.x_train = Variable(data.x[self.ind_train,:,:], requires_grad=False)	#n_user x len_seq x n_feat
		self.x_test = Variable(data.x[self.ind_test,:,:], requires_grad=False)	#n_user x len_seq x n_feat
		self.y_train = Variable(data.y[self.ind_train,:], requires_grad=False)	#n_user x n_soft
		self.y_test = Variable(data.y[self.ind_test,:], requires_grad=False)	#n_user x n_soft
		self.z_train = Variable(data.z[torch.LongTensor(self.ind_train)], requires_grad=False)	#n_user
		self.z_test = Variable(data.z[torch.LongTensor(self.ind_test)], requires_grad=False)	#n_user		

	def train(self):
		self.model = LstmNet(self.params)
		self.losses_train = []
		self.losses_test = []
		for _ in tqdm(xrange(self.params['n_epoch']), ncols=100):		
			loss_train = self.model.get_loss(self.x_train, self.y_train, self.z_train)
			#print(torch.stack((self.z_test, self.model.Z_pred.squeeze())))
			self.model.optimizer.zero_grad()
			loss_train.backward()
			self.model.optimizer.step()
			self.losses_train.append(loss_train.data[0])

			loss_test = self.model.get_loss(self.x_test, self.y_test, self.z_test)
			self.losses_test.append(loss_test.data[0])

	def evaluate(self):
		y_pred_train, z_pred_train = self.model.forward(self.x_train)
		correct_train = map(lambda i: (z_pred_train.data.numpy()[i] > 0.5 and int(self.z_train.data.numpy()[i] == 1)) \
			or (z_pred_train.data.numpy()[i] <= 0.5 and int(self.z_train.data.numpy()[i] == 0)), range(z_pred_train.size()[0]))
		print('Training accuracy: '+str(correct_train.count(True)*1.0/len(correct_train)))

		y_pred_test, z_pred_test = self.model.forward(self.x_test)
		correct_test = map(lambda i: (z_pred_test.data.numpy()[i] > 0.5 and int(self.z_test.data.numpy()[i] == 1)) \
			or (z_pred_test.data.numpy()[i] <= 0.5 and int(self.z_test.data.numpy()[i] == 0)), range(z_pred_test.size()[0]))
		print('Testing accuracy: '+str(correct_test.count(True)*1.0/len(correct_test)))
		tp = map(lambda i: (z_pred_test.data.numpy()[i] > 0.5 and int(self.z_test.data.numpy()[i] == 1)), range(z_pred_test.size()[0])).count(True)
		tn = map(lambda i: (z_pred_test.data.numpy()[i] <= 0.5 and int(self.z_test.data.numpy()[i] == 0)), range(z_pred_test.size()[0])).count(True)
		fp = map(lambda i: (z_pred_test.data.numpy()[i] > 0.5 and int(self.z_test.data.numpy()[i] == 0)), range(z_pred_test.size()[0])).count(True)
		fn = map(lambda i: (z_pred_test.data.numpy()[i] <= 0.5 and int(self.z_test.data.numpy()[i] == 1)), range(z_pred_test.size()[0])).count(True)
		if tp+fp > 0:
			print('Testing precision: '+str(tp*1.0/(tp+fp)))
		else:
			print('Testing precision divided by zero')
		if tp+fn > 0:
			print('Testing recall: '+str(tp*1.0/(tp+fn)))
		else:
			print('Testing recall divided by zero')

		tp_sub = [0]*self.params['n_type']
		fp_sub = [0]*self.params['n_type']
		fn_sub = [0]*self.params['n_type']
		for i in range(y_pred_test[0].size()[0]):
			pred_list = list(map(lambda k: k.data[i].numpy()[0], y_pred_test))
			truth_list = list(self.y_test.data[i].numpy())
			pred_sub = pred_list.index(max(pred_list))
			truth_sub = truth_list.index(max(truth_list))
			if pred_sub == truth_sub:
				tp_sub[pred_sub] += 1
			else:
				fp_sub[pred_sub] += 1
				fn_sub[truth_sub] += 1

		for i in range(len(tp_sub)):
			if tp_sub[i]+fp_sub[i] > 0:
				print('Testing precision - subtype '+str(i)+' :'+str(tp_sub[i]*1.0/(tp_sub[i]+fp_sub[i])))
			else:
				print('Testing precision - subtype '+str(i)+' divided by zero')
			if tp_sub[i]+fn_sub[i] > 0:
				print('Testing recall - subtype '+str(i)+' :'+str(tp_sub[i]*1.0/(tp_sub[i]+fn_sub[i])))
			else:
				print('Testing recall - subtype '+str(i)+' divided by zero')

		matplotlib.rcParams['pdf.fonttype'] = 42
		matplotlib.rcParams['ps.fonttype'] = 42
		length = len(self.losses_train)
		plt.plot(np.array(range(length)), self.losses_train, label='train')
		plt.plot(np.array(range(length)), self.losses_test, label='test')
		plt.xlabel('Epoch',fontsize=15)
		plt.ylabel('Loss',fontsize=15)
		plt.grid()
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.legend(fontsize=12)
		plt.savefig('../plots/losses.png', format='png', dps=200, bbox_inches='tight')

if __name__ == '__main__':
	params = {}
	params['access_day'] = (0, 5)
	params['access_feat'] = (0, 12)
	params['data_x'] = '../data/act_data.pkl'
	params['data_y'] = '../data/soft.pkl'
	params['data_z'] = '../data/churns.pkl'
	params['emb_size'] = 64
	params['emb_size_front'] = 32
	params['n_type'] = 6
	params['n_layer'] = 2
	params['dropout'] = 0.5
	params['lambda'] = 1
	params['learning_rate'] = 0.1
	params['n_epoch'] = 100
	params['split_ratio'] = 0.8

	predictor = Predictor(params)
	predictor.build_input()
	predictor.train()
	predictor.evaluate()

	
