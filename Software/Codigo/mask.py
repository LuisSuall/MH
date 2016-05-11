import numpy as np

class Mask:

	def __init__(self,length):
		self.values = np.zeros(length, dtype = bool)
		self.score = 0

	def randomize(self):
		self.values = np.random.choice([True,False], len(self.values),
									   replace = True)

	def get(self,i):
		return self.values[i]

	def set(self,i,value):
		self.values[i] = value
		
	def get_score(self):
		return self.score

	def set_score(self,score):
		self.score = score

	def length(self):
		return len(self.values)

	def flip(self,i):
		if self.values[i]:
			self.values[i] = False
		else:
			self.values[i] = True

	def set_false(self):
		self.values = np.zeros(self.length(), dtype = bool)

	def mutate(self,s):
		mutate_idx = np.random.choice(range(self.length()),s,replace=False)

		for idx in mutate_idx:
			self.flip(idx)
