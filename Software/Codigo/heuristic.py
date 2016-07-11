import classifier
import random
from math import log, exp
from mask import Mask
import numpy as np
import math

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

class KNNHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self,mask,max_iter):
		mask.set_true()
		return self.classifier.score_train(mask)

class BMBHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self,mask,max_iter):
		LS = LSHeuristic(self.classifier)

		best_score = 0
		best_mask = Mask(mask.length())

		for _ in range(25):
			mask.randomize()
			score = LS.run(mask,max_iter)

			if score > best_score:
				best_score = score
				best_mask.values = np.copy(mask.values)

		mask.values = np.copy(best_mask.values)
		return best_score

class ASFSHeuristic(Heuristic):
	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self,mask, tolerance = 0.3):
		best_score = 0
		change_produced = True

		while(change_produced):
			neighbourhood_best_score = 0
			neighbourhood_worst_score = 100
			change_produced = False
			saved_scores = np.full((mask.length(),2),-100,dtype = np.float32)

			for idx in range(0,mask.length()):
				if not mask.get(idx):
					score = self.classifier.score_train(mask, idx)
					saved_scores[idx] = [idx,score]
					if  score > neighbourhood_best_score:
						neighbourhood_best_score = score
					elif score < neighbourhood_worst_score:
						neighbourhood_worst_score = score

			min_accepted = neighbourhood_best_score - tolerance * (neighbourhood_best_score - neighbourhood_worst_score)
			saved_scores = saved_scores[saved_scores[:,1]>min_accepted]
			selected_neighbour = saved_scores[random.randint(0,len(saved_scores)-1)]

			if selected_neighbour[1] > best_score:
				change_produced = True
				best_score = selected_neighbour[1]
				mask.flip(selected_neighbour[0])

		return best_score

class GRASPHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self,mask,max_iter):
		LS = LSHeuristic(self.classifier)
		ASFS = ASFSHeuristic(self.classifier)

		best_score = 0
		best_mask = Mask(mask.length())

		for _ in range(25):
			mask.set_false()
			score_asfs = ASFS.run(mask)
			score = LS.run(mask,max_iter)

			if score > best_score:
				best_score = score
				best_mask.values = np.copy(mask.values)

		mask.values = np.copy(best_mask.values)
		return best_score

class ILSHeuristic(Heuristic):

	def __init__(self,classifier):
		super().__init__(classifier)

	def run(self,mask,max_iter):
		LS = LSHeuristic(self.classifier)
		s = round(mask.length()*0.1)

		best_score = LS.run(mask,max_iter)
		best_mask = Mask(mask.length())
		best_mask.values = np.copy(mask.values)

		for _ in range(24):
			mask.mutate(s)
			score = LS.run(mask,max_iter)

			if score > best_score:
				best_score = score
				best_mask.values = np.copy(mask.values)
			else:
				mask.values = np.copy(best_mask.values)

		mask.values = np.copy(best_mask.values)
		return best_score

class GenericGeneticHeuristic(Heuristic):
	def __init__(self,classifier):
		super().__init__(classifier)

	def random_population(self,size_population,mask_size):
		population = []
		for _ in range(size_population):
			mask = Mask(mask_size)
			mask.randomize()
			population.append(mask)

		return population

	def select_parents(self,current_gen,num_parents):
		mask_size = current_gen[0].length()
		idx_parents = np.random.choice(range(len(current_gen)),
									   num_parents*2,
									   replace = True)
		parents = []

		for idx in range(num_parents):
			parent_idx = -1
			if current_gen[idx_parents[idx*2]].get_score() > current_gen[idx_parents[idx*2+1]].get_score():
				parent_idx = idx_parents[idx*2]
			else:
				parent_idx = idx_parents[idx*2+1]

			mask = Mask(mask_size)
			mask.values = np.copy(current_gen[parent_idx].values)
			parents.append(mask)

		return parents

	def cross_pair(self,first_parent, second_parent):
		first_son = Mask(first_parent.length())
		second_son = Mask(first_parent.length())

		for idx in range(first_parent.length()):
			if first_parent.get(idx) == second_parent.get(idx):
				first_son.set(idx,first_parent.get(idx))
				second_son.set(idx,first_parent.get(idx))
			else:
				value = random.sample([True,False],1)[0]
				first_son.set(idx,value)
				second_son.set(idx,not value)

		return first_son, second_son

	def cross(self,next_gen,prob_cross):
		num_crosses = round(len(next_gen)*prob_cross/2)

		for idx in range(num_crosses):
			first_son, second_son = self.cross_pair(next_gen[idx*2],next_gen[idx*2+1])
			next_gen[idx*2] = first_son
			next_gen[idx*2+1] = second_son

	def mutate(self,next_gen,prob_mutation):
		num_gen = len(next_gen)
		size_mask = next_gen[0].length()

		num_mutations = math.floor(num_gen*size_mask*prob_mutation)
		if num_mutations == 0:
			if random.uniform(0,1) < num_gen*size_mask*prob_mutation:
				num_mutations = 1

		idx_mutation = np.random.choice(range(num_gen*size_mask),
									   num_mutations,
									   replace = True)

		for idx in idx_mutation:
			next_gen[idx//size_mask].flip(idx%size_mask)

	def run(self,mask,max_iter,num_parents,replace_gen,prob_cross,size_population=30,prob_mutation = 0.001):
		current_gen = self.random_population(size_population,mask.length())
		num_evals = 0

		for mask in current_gen:
			num_evals += 1
			mask.set_score(self.classifier.score_train(mask))

		current_gen = sorted(current_gen, key = lambda x:x.get_score(), reverse = True)

		while num_evals < max_iter:
			next_gen = self.select_parents(current_gen,num_parents)
			self.cross(next_gen,prob_cross)
			self.mutate(next_gen,prob_mutation)

			for mask in next_gen:
				num_evals += 1
				mask.set_score(self.classifier.score_train(mask))
			next_gen = sorted(next_gen, key = lambda x:x.get_score(), reverse = True)

			replace_gen(current_gen,next_gen)
			current_gen = sorted(current_gen, key = lambda x:x.get_score(), reverse = True)

		mask = current_gen[0]
		return mask.get_score()


class AGEHeuristic(Heuristic):
	def __init__(self,classifier):
		super().__init__(classifier)

	def replace_gen(self,current_gen,next_gen):
		if current_gen[-2].get_score() < next_gen[0].get_score() and current_gen[-2].get_score() < next_gen[1].get_score():
			current_gen[-2] = next_gen[0]
			current_gen[-1]  = next_gen[1]
		elif current_gen[-1].get_score() < next_gen[0].get_score():
			current_gen[-1] = next_gen[0]

	def run(self,mask,max_iter):
		GGHeuristic = GenericGeneticHeuristic(self.classifier)
		return GGHeuristic.run(mask,max_iter,2,self.replace_gen,1)

class AGGHeuristic(Heuristic):
	def __init__(self,classifier):
		super().__init__(classifier)

	def replace_gen(self,current_gen,next_gen):
		if current_gen[0].get_score() > next_gen[0].get_score():
			next_gen[-1] = current_gen[0]

	def run(self,mask,max_iter):
		GGHeuristic = GenericGeneticHeuristic(self.classifier)
		return GGHeuristic.run(mask,max_iter,30,self.replace_gen,0.7)

class MemeticHeuristic(GenericGeneticHeuristic):
	def __init__(self,classifier,num_LS,elitism_LS):
		super().__init__(classifier)
		self.num_LS = num_LS
		self.elitism_LS = elitism_LS

	def replace_gen(self,current_gen,next_gen):
		if current_gen[0].get_score() > next_gen[0].get_score():
			next_gen[-1] = current_gen[0]

	def memetic_LS(self,current_gen,num_LS,elitism_LS):
		num_evals = 0

		if elitism_LS:
			gen_to_LS = range(num_LS)
		else:
			gen_to_LS = np.random.choice(range(len(current_gen)),
										   num_LS,
										   replace = False)

		for idx_gen in gen_to_LS:
			mask = current_gen[idx_gen]
			best_score = mask.get_score()

			for idx in random.sample(range(mask.length()),mask.length()):
				score = self.classifier.score_train(mask, idx)
				num_evals += 1
				if  score > best_score:
					mask.flip(idx)
					mask.set_score(score)
					break

		return num_evals

	def run(self,mask,max_iter,prob_cross=0.7,num_parents=10,size_population=10,prob_mutation = 0.001):
		current_gen = self.random_population(size_population,mask.length())
		num_evals = 0
		gen_number = 0

		for mask in current_gen:
			num_evals += 1
			mask.set_score(self.classifier.score_train(mask))

		current_gen = sorted(current_gen, key = lambda x:x.get_score(), reverse = True)

		while num_evals < max_iter:
			gen_number += 1

			print("Generación: " + str(gen_number))
			print("Número evaluaciones: " + str(num_evals))

			next_gen = self.select_parents(current_gen,num_parents)
			self.cross(next_gen,prob_cross)
			self.mutate(next_gen,prob_mutation)

			for mask in next_gen:
				num_evals += 1
				mask.set_score(self.classifier.score_train(mask))
			next_gen = sorted(next_gen, key = lambda x:x.get_score(), reverse = True)

			self.replace_gen(current_gen,next_gen)

			current_gen = sorted(next_gen, key = lambda x:x.get_score(), reverse = True)

			if gen_number % 10 == 0:
				print("He entrado en el LS.")
				num_evals += self.memetic_LS(current_gen,self.num_LS,self.elitism_LS)
				print("num_evals: " + str(num_evals))

			current_gen = sorted(current_gen, key = lambda x:x.get_score(), reverse = True)

		mask = current_gen[0]
		return mask.get_score()
