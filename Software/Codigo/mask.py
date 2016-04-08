import numpy as np

class Mask:

	def __init__(self,length):
		self.values = np.zeros(length, dtype = bool)

	def randomize(self):
		self.values = np.random.choice([True,False], len(self.values),
									   replace = True)

	def get(self,i):
		return self.values[i]

	def flip(self,i):
		if self.values[i]:
			self.values[i] = False
		else:
			self.values[i] = True

	def length(self):
		return len(self.values)
