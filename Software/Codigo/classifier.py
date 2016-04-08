import numpy as np
from sklearn import neighbors,cross_validation

class Classifier:

	def __init__(self, train_data, train_label, test_data, test_label):
		self.__train_data = train_data
		self.__train_label = train_label
		self.__test_data = test_data
		self.__test_label = test_label
		self.__knn = neighbors.KNeighborsClassifier(3, weights = 'distance')

	def score_train(self, mask, i = None):

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

	def score_test(self, mask):
		self.__knn.fit(self.__train_data[:,mask.values], self.__train_label)
		return self.__knn.score(self.__test_data[:,mask.values], self.__test_label)
