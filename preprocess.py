import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path

class PreProcess():

	def __init__(self):
		self.user_act = {}

	def msg_load(self, path="../data/msg/"):
		dates = [str(i) for i in range(1, 32)]
		msgs = [[] for i in range(31)]
		for i in range(31):
  			if len(dates[i]) == 1:
  				dates[i] = '0' + dates[i]
  			with open(path+'msg_sampled_201708'+dates[i]) as f:
  				for line in f:
  					words = line.split(',')
  					if words[0] == 'userId':
  						tokens=words[:]
  					else:
  						t = {}
  						for j in range(len(tokens)):
  							t[tokens[j].strip()] = words[j].strip()
  						msgs[i].append(t)
  		
		for i in range(31):
			for t in msgs[i]:
				if t['userId'] not in self.user_act:
					self.user_act[t['userId']] = np.zeros(shape=(15,14), dtype=np.float32)
				create = int(t['creationDay'][-2:])
				day = i+1-create
				if 0 <= day and day <= 14:
					self.user_act[t['userId']][day,0:5] = [int(t['chat_received']), int(t['chat_sent']), int(t['snap_viewed']), int(t['snap_sent']), int(t['story_view'])]      
		print('msg users: ' + str(len(self.user_act)))

	def msg_vis(self, token='chat_received'):
		if len(self.user_act) == 0:
			self.msg_load()
		for i in self.user_act.keys():
			if token == 'chat_received':
				y = np.array(self.user_act[i][:, 0])
			elif token == 'chat_sent':
				y = np.array(self.user_act[i][:, 1])
			elif token == 'snap_viewed':
				y = np.array(self.user_act[i][:, 2])
			elif token == 'snap_sent':
				y = np.array(self.user_act[i][:, 3])
			else:
				y = np.array(self.user_act[i][:, 4])
			x = np.array(range(15))
			plt.plot(x, y, linewidth=0.5)
		plt.show()

	def dis_load(self, path="../data/discover/"):
		dates = [str(i) for i in range(1, 32)]
		msgs = [[] for i in range(31)]
		for i in range(31):
  			if len(dates[i]) == 1:
  				dates[i] = '0' + dates[i]
  			with open(path+'discover_sampled_201708'+dates[i]) as f:
  				for line in f:
  					words = line.split(',')
  					if words[0] == 'userId':
  						tokens=words[:]
  					else:
  						t = {}
  						for j in range(len(tokens)):
  							t[tokens[j].strip()] = words[j].strip()
  						msgs[i].append(t)
  
  		user_dis = []
		for i in range(31):
			for t in msgs[i]:
				if t['userId'] not in self.user_act:
					self.user_act[t['userId']] = np.zeros(shape=(15,14), dtype=np.float32)
				if t['userId'] not in user_dis:
					user_dis.append(t['userId'])
				create = int(t['creationDay'][-2:])
				day = i+1-create
				if 0 <= day and day <= 14:
					if len(t['discover_view_count']) > 0:
						self.user_act[t['userId']][day,5] += int(t['discover_view_count'])
		print('union, discover users: ' + str((len(self.user_act), len(user_dis))))


	def dis_vis(self):
		if len(self.user_act) == 0:
			self.dis_load()
		for i in self.user_act.keys():
			y = np.array(self.user_act[i][:, 5])
			x = np.array(range(15))
			plt.plot(x, y, linewidth=0.5)
		plt.show()

	def lens_load(self, path="../data/lens/"):
		dates = [str(i) for i in range(1, 32)]
		msgs = [[] for i in range(31)]
		for i in range(31):
  			if len(dates[i]) == 1:
  				dates[i] = '0' + dates[i]
  			with open(path+'lens_sampled_201708'+dates[i]) as f:
  				for line in f:
  					words = line.split(',')
  					if words[0] == 'userId':
  						tokens=words[:]
  					else:
  						t = {}
  						for j in range(len(tokens)):
  							t[tokens[j].strip()] = words[j].strip()
  						msgs[i].append(t)
  
  		user_lens = []
		for i in range(31):
			for t in msgs[i]:
				if t['userId'] not in self.user_act:
					self.user_act[t['userId']] = np.zeros(shape=(15,14), dtype=np.float32)
				if t['userId'] not in user_lens:
					user_lens.append(t['userId'])
				create = int(t['creationDay'][-2:])
				day = i+1-create
				if 0 <= day and day <= 14:
					self.user_act[t['userId']][day,6:10] = [int(t['post_count']), int(t['sent_count']), int(t['save_count']), int(t['swipe_count'])]
		print('union, lens users: ' + str((len(self.user_act), len(user_lens))))


	def lens_vis(self, token='post_count'):
		if len(self.user_act) == 0:
			self.lens_load()
		for i in self.user_lens.keys():
			if token == 'post_count':
				y = np.array(self.user_act[i][:, 6])
			elif token == 'sent_count':
				y = np.array(self.user_act[i][:, 7])
			elif token == 'save_count':
				y = np.array(self.user_act[i][:, 8])
			else:
				y = np.array(self.user_act[i][:, 9])
			x = np.array(range(15))
			plt.plot(x, y, linewidth=0.5)
		plt.show()


	def link_load(self, path="../data/link/"):
		linklist = []
		userlist = []
		with open(path+'linkset_sampled', 'r') as f:
			for line in f:
  				words = line.split(',')
  				if words[0] == 'fromUserID':
  					tokens=words[:]
  				else:
  					t = {}
  					for j in range(len(tokens)):
  						t[tokens[j].strip()] = words[j].strip()
  					linklist.append(t)
  		with open(path+'userset_sampled', 'r') as f:
  			for line in f:
  				words = line.split(',')
  				if words[0] == 'userId':
  					tokens=words[:]
  				else:
  					t = {}
  					for j in range(len(tokens)):
  						t[tokens[j].strip()] = words[j].strip()
  					userlist.append(t)

		users = []
		original_link = {}
		added_link = [{} for i in range(31)]
		for link in linklist:
			if link['fromUserID'] not in users:
				users.append(link['fromUserID'])
			if link['toUserID'] not in users:
				users.append(link['toUserID'])
			u1 = users.index(link['fromUserID'])
			u2 = users.index(link['toUserID'])
			if u1 == u2:
				continue
			if link['addedAt'][0:7] != '2017-08':
				if u1 not in original_link:
					original_link[u1] = set()
				original_link[u1].add(u2)
				if u2 not in original_link:
					original_link[u2] = set()
				original_link[u2].add(u1)
			else:
				day = int(link['addedAt'][-2:])-1
				if u1 not in added_link[day]:
					added_link[day][u1] = set()
				added_link[day][u1].add(u2)
				if u2 not in added_link[day]:
					added_link[day][u2] = set()
				added_link[day][u2].add(u1)

		user_link = []
		print(len(userlist))
		pos = 0
		for user in userlist:
			print(pos)
			pos += 1
			userid = user['userId']
			if userid not in users:
				continue
			u = users.index(userid)
			creationDay = int(user['CreationDay'][-2:])-1
			if userid not in self.user_act:
				self.user_act[userid] = np.zeros(shape=(15,14), dtype=np.float32)
			if userid not in user_link:
				user_link.append(userid)
			edges = set()
			nodes = set()
			nodes.add(u)
			degree = 0
			density = 1
			cc = 1  #clustering coefficient
			bc = 1  #betweenness centrality
  
			for i in range(15):
				day = i+creationDay
				if u in added_link[day]:
					links = added_link[day][u]
					newnodes = set()
					for new in links:
						nodes.add(new)
					for new in links:
						for old in nodes:
							if new in original_link and old in original_link[new]:
								edges.add((new, old))
								edges.add((old, new))
							for t in range(day):
								if new in added_link[t] and old in added_link[t][new]:
									edges.add((new, old))
									edges.add((old, new))
				for u1 in nodes:
					for u2 in nodes:
						if u1 != u2 and u1 in added_link[day] and u2 in added_link[day][u1]:
							edges.add((u1, u2))
							edges.add((u2, u1))
				degree = len(nodes)-1
				if len(nodes) > 1:
					density = len(edges)*1.0/(len(nodes)*(len(nodes)-1))

				triplet = 0
				triangle = 0
				for u1 in nodes:
					for u2 in nodes:
						for u3 in nodes:
							if u1 != u2 and u2 != u3 and u1 != u3:
								if (u1, u2) in edges and (u1, u3) in edges:
									triplet += 1
									if (u2, u3) in edges:
										triangle += 1
				if triplet > 0:
					cc = triangle*1.0/triplet
				npairs = (len(nodes)-1)*(len(nodes)-2)
				nb = 0
				for u1 in nodes:
					for u2 in nodes:
						if u1 != u and u2 != u and u1 != u2 and (u1, u2) not in edges:
							nb += 1
				if npairs > 0:
					bc = nb*1.0/npairs
				self.user_act[userid][i,10:14] = [degree, density, cc, bc]

			for dim in xrange(10, 14):
				self.user_act[userid][:, dim] = np.diff(np.insert(self.user_act[userid][:, dim], 0, 0))

		print('union, linked users: ' + str((len(self.user_act), len(user_link))))

	def link_vis(self):
		try:
			self.link_measures
		except AttributeError:
			self.link_load()
		x = np.array(range(15))
		for i in xrange(4):
			for j in xrange(len(self.link_measures)):
				y = self.link_measures[j][:,i]
				plt.plot(x, y, linewidth=0.5)
			if i == 0:
				plt.axis([0, 14, 0, 160])
			else:
				plt.axis([0, 14, 0, 1])
			plt.savefig('link_measures_'+str(i)+'.png')
			plt.clf()

	def act_store(self, file='../data/act_data.pkl'):
		if len(self.user_act) == 0:
			self.msg_load()
			self.dis_load()
			self.lens_load()
			self.link_load()
		useract = []
		userkey = []
		for key in self.user_act.keys():
			if np.max(np.sum(self.user_act[key], axis=0)) > 0:
				useract.append(self.user_act[key])
				userkey.append(key)
		self.user_act = useract
		with open(file, 'wb') as f:
			pickle.dump(useract, f)
			pickle.dump(userkey, f)
		print(len(self.user_act))
		#each user corresponds to a matrix
		#each row is the activity/measure data of one day
		#each column is the activity/measure of one type


	def act_load(self, file='../data/act_data.pkl'):
		with open(file, 'rb') as f:
			self.user_act = pickle.load(f)

	def corr_ana(self):
		n_user = len(self.user_act)
		act_data = []
		cor = []
		for act in range(14):
			act_data += [np.empty(shape=(n_user, 15), dtype=np.float32)]
			cor += [[0] * 14]
		for user in range(len(self.user_act)):
			for act in range(14):
				act_data[act][user, :] = self.user_act[user][:, act]
		for i in range(14):
			act_data[i] /= np.max(act_data[i])
		for i in range(14):
			for j in range(14):
				cor[i][j] = np.linalg.norm(act_data[i]-act_data[j])
		plt.imshow(cor, cmap='Blues', interpolation='nearest')
		plt.savefig("../plots/simple_cross.png", bbox_inches='tight')


if __name__ == '__main__':
	pre = PreProcess()

	if not os.path.exists('../data/act_data.pkl'):
		pre.act_store()
	else:
		pre.act_load()

	pre.corr_ana()

#pre.msg_vis()
#pre.dis_vis()
#pre.lens_vis()
#pre.link_vis()
