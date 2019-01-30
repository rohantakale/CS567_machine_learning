import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		pred_sign = [0] * len(features)

		for i in range(len(self.clfs_picked)):
			clfs_pred = self.clfs_picked[i].predict(features)
			#print('clfs preds = ', clfs_pred)
			for j in range(len(features)):
				pred_sign[j] += self.betas[i] * clfs_pred[j]

		for k in range(len(pred_sign)):
			if pred_sign[k] < 0:
				pred_sign[k] = -1
			else:
				pred_sign[k] = 1
				
		return pred_sign
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		wts = np.ones(N)* (1/N)

		for i in range(self.T):
			eta = []
			for i in range(len(list(self.clfs))):
				clfs_pred = list(self.clfs)[i].predict(features)
				eta.append(np.dot(wts,np.not_equal(labels,clfs_pred)))
			min_eta = min(eta)
			h_t = eta.index(min_eta)

			filt_1 = [int(k) for k in np.not_equal(labels,list(self.clfs)[h_t].predict(features))]
			predictn = [sgn if sgn == 1 else -1 for sgn in filt_1]

			self.clfs_picked.append(list(self.clfs)[h_t])
			beta = 0.5 * np.log((1 - min_eta)/min_eta)
			self.betas.append(beta)

			wts = np.multiply(wts, np.exp([sign1 * beta for sign1 in predictn]))
			wts = wts / np.sum(wts)

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	