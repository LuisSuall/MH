import classifier
import random
from math import log, exp
from mask import Mask
import numpy as np

class Heuristic:

	def __init__(self,classifier = None):
		self.classifier = classifier

	def set_classifier(self,classifier):
		self.classifier = classifier

	def run(self,mask, max_iter):
		pass


class GreedyHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self, mask, max_iter):
		best_idx = -1
		best_score = 0
		change_produced = True

		while(change_produced):
			change_produced = False
			for idx in range(0,mask.length()):
				if not mask.get(idx):
					score = self.classifier.score_train(mask, idx)
					if  score > best_score:
						best_score = score
						best_idx = idx
						change_produced = True

			if change_produced:
				mask.flip(best_idx)

		return best_score




class LSHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self, mask, max_iter):
		num_sol = 0
		best_score = self.classifier.score_train(mask)
		changed = True

		while changed and num_sol < max_iter:
			changed = False

			for idx in random.sample(range(mask.length()),mask.length()):
				score = self.classifier.score_train(mask, idx)
				num_sol += 1
				if  score > best_score:
					best_score = score
					mask.flip(idx)
					changed = True
					break

		return best_score

class SAHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self, mask, max_iter):
		num_sol = 0
		best_score = self.classifier.score_train(mask)
		best_mask = Mask(mask.length())
		best_mask.values = np.copy(mask.values)
		mu = 0.3
		phi = 0.3
		data_length = mask.length()

		max_sol_inner_loop = 10 * data_length
		max_accepted_sol = 0.1 * max_sol_inner_loop

		temp =(mu*best_score)/(-log(phi))
		end_temp = 0.003
		beta = (temp - end_temp)/((max_iter/max_sol_inner_loop)*temp*end_temp)
		accepted_sol = 1

		while temp >= end_temp and num_sol < max_iter and accepted_sol != 0:
			accepted_sol = 0
			for cont in range(max_sol_inner_loop):
				if(accepted_sol >= max_accepted_sol):
					break

				idx = random.randint(0,data_length-1)
				score = self.classifier.score_train(mask,idx)
				num_sol += 1

				if score > best_score:
					best_score = score
					mask.flip(idx)
					best_mask.values = np.copy(mask.values)
					accepted_sol += 1
					continue

				delta = best_score-score

				if delta == 0:
					continue

				if random.uniform(0,1) <= exp(-(delta/temp)):
					mask.flip(idx)
					accepted_sol += 1

			temp = temp/(1+beta*temp)

		mask.values = np.copy(best_mask.values)
		return best_score


class TABUHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self, mask, max_iter):
		num_sol = 0
		best_score = self.classifier.score_train(mask)
		best_mask = Mask(mask.length())
		best_mask.values = np.copy(mask.values)

		data_length = mask.length()
		tabu_list = [-1] * (data_length//3)

		best_idx = -1
		best_neighbourhood_score = 0
		while num_sol < max_iter:
			best_idx = -1
			best_neighbourhood_score = 0

			for idx in random.sample(range(mask.length()),30):
				score = self.classifier.score_train(mask,idx)
				num_sol += 1

				if score > best_score:
					best_score = score
					best_mask.values = np.copy(mask.values)
					best_mask.flip(idx)

					best_neighbourhood_score = score
					best_idx = idx
					continue

				if score > best_neighbourhood_score:
					if not idx in tabu_list:
						best_neighbourhood_score = score
						best_idx = idx


			mask.flip(best_idx)
			tabu_list.pop(0)
			tabu_list.append(best_idx)

		mask.values = np.copy(best_mask.values)
		return best_score

class BMBHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(mask):
		LS = LSHeuristic(self.classifier)

		best_score = 0
		best_mask = Mask(mask.length())

		for _ in range(25):
			mask.randomize()
			score = LS.run(mask)

			if score > best_score:
				best_score = score
				best_mask.values = np.copy(mask.values)

		mask.values = np.copy(best_mask.values)
		return best_score

class ASFSHeuristic(Heuristic):
	def __init__(self,classifier):
		super().__init__(classifier)

	def run(mask, tolerance = 0.3):
		best_score = 0
		change_produced = True

		while(change_produced):
			neighbourhood_best_score = 0
			neighbourhood_worst_score = 100
			change_produced = False
			saved_scores = np.full((mask.length(),2),-100,dtype = float64)

			for idx in range(0,mask.length()):
				if not mask.get(idx):
					score = self.classifier.score_train(mask, idx)
					saved_scores[idx] = [idx,score]
					if  score > best_score:
						neighbourhood_best_score = score
					elif score < worst_score:
						neighbourhood_worst_score = score

			min_accepted = best_score - tolerance * (best_score - worst_score)
			saved_scores = saved_scores[saved_scores[:,1]>min_accepted]
			selected_neighbour = saved_scores[random.randint(0,len(saved_scores))]

			if selected_neighbour[1] > best_score:
				change_produced = True
				best_score = selected_neighbour[1]
				mask.flip(selected_neighbour[0])

		return best_score

class GRASPHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(mask):
		LS = LSHeuristic(self.classifier)
		ASFS = ASFSHeuristic(self.classifier)

		best_score = 0
		best_mask = Mask(mask.lenght)

		for _ in range(25):
			mask.set_false()
			ASFS.run(mask)
			score = LS.run(mask)

			if score > best_score:
				best_score = score
				best_mask.values = np.copy(mask.values)

		mask.values = np.copy(best_mask.values)
		return best_score

class ILSHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(mask):
		LS = LSHeuristic(self.classifier)
		s = round(mask.lenght*0.1)

		best_score = LS.run(mask)
		best_mask = Mask(mask.lenght)
		best_mask.values = np.copy(mask.values)

		for _ in range(24):
			mask.mutate(s)
			score = LS.run(mask)

			if score > best_score:
				best_score = score
				best_mask.values = np.copy(mask.values)
			else:
				mask.values = np.copy(best_mask.values)

		mask.values = np.copy(best_mask.values)
		return best_score
