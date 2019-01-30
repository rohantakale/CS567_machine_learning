import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branch_arr = np.array(branches)
			branch_T = np.transpose(branch_arr).tolist()
			total = float(np.sum(branch_arr)) # 8
			cd_entropy = 0.0

			for b in branch_T:
			    br_total = float(sum(b)) # 6 and then for = 2
			    weight = br_total / total
			    if br_total == 0:	# can't divide by zero so go to next iteration
			        continue
			    for cls_val in b:
			    	if cls_val > 0:
			    	    prob = float(cls_val) / br_total # 2/6 and then 2/2
			    	    cd_entropy -= prob * np.log(prob) * weight
			return cd_entropy

		if not self.splittable:
			return

		features = np.array(self.features)
		entr = []

		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			feat = features[:, idx_dim]
			divn = []

			for t in np.unique(feat):
				t_features = feat[np.where(feat == t)]
				t_labels = np.array(self.labels)[np.where(feat == t)]
				bran = []

				for i in range(self.num_cls):
					bran.append(np.sum(t_labels == i))
				divn.append(bran)

			entr.append(conditional_entropy(np.array(divn).T.tolist()))

		self.dim_split = np.argmin(entr)
		feat  = features[:,self.dim_split]
		self.feature_uniq_split = np.unique(feat).tolist()
		#print('self.feature unique = ',self.feature_uniq_split)

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		if len(np.unique(feat)) > 1:
			for t in np.unique(feat):
				t_features = features[np.where(feat == t)].tolist()
				t_labels = np.array(self.labels)[np.where(feat == t)].tolist()
				self.children.append(TreeNode(t_features, t_labels, self.num_cls))
		else:
			self.splittable = False

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



