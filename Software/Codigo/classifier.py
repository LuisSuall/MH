import numpy as np
from sklearn import neighbors,cross_validation
from multiprocessing import Pool
from functools import partial
from knnGPU.knnLooGPU import knnLooGPU

class Classifier:

	def __init__(self, train_data, train_label, test_data, test_label):
		self.__train_data = train_data
		self.__train_label = train_label
		self.__test_data = test_data
		self.__test_label = test_label
		self.__knn = neighbors.KNeighborsClassifier(3, weights = 'distance')
		self.cuda_knn = knnLooGPU(train_data.shape[0],test_data.shape[0],train_data.shape[1],3)

	def score_train(self, mask, i = None):
		if i != None:
			mask.flip(i)

		X = self.__train_data[:,mask.values]
		y = self.__train_label

		if X.size != 0:
			score = self.cuda_knn.scoreSolution(X,y)
		else:
			score = 0
			
		if i != None:
			mask.flip(i)

		return score

	def score_test(self, mask):
		train_data = self.__train_data[:,mask.values]
		test_data = self.__test_data[:,mask.values]
		train_label = self.__train_label
		test_label = self.__test_label

		score = self.cuda_knn.scoreOut(train_data,test_data,train_label,test_label)

		return score

	def score_train_cpu(self, mask, i = None):

		if i != None:
			mask.flip(i)

		X = self.__train_data[:,mask.values]
		y = self.__train_label
		loo = cross_validation.LeaveOneOut(len(y))

		score = 0

		for train_index, test_index in loo:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			self.__knn.fit(X_train, y_train)
			score += self.__knn.score(X_test,y_test)

		if i != None:
			mask.flip(i)

		return 100*score / len(y)

	def score_test_cpu(self, mask):
		self.__knn.fit(self.__train_data[:,mask.values], self.__train_label)
		return self.__knn.score(self.__test_data[:,mask.values], self.__test_label)
