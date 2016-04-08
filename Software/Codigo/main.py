import time
import heuristic, mask, data_handler, classifier
import numpy as np
import random
import csv
import sys

def main():
	if len(sys.argv) > 2:

		data_hand = data_handler.DataHandler()
		if sys.argv[1] == "arrhythmia":
			data_hand.load_arrhythmia()
		elif sys.argv[1] == "libras":
			data_hand.load_libras()
		elif sys.argv[1] == "wdbc":
			data_hand.load_wdbc()
		else:
			print("Opcion de BD incorrecta")
			return

		if sys.argv[2] == "SFS":
			my_heuristic = heuristic.GreedyHeuristic(None)
		elif sys.argv[2] == "LS":
			my_heuristic = heuristic.LSHeuristic(None)
		elif sys.argv[2] == "SA":
			my_heuristic = heuristic.SAHeuristic(None)
		elif sys.argv[2] == "TS":
			my_heuristic = heuristic.TABUHeuristic(None)
		else:
			print("Opcion de BD incorrecta")
			return

		seeds = [12345678,90123456,78901234,456789012,34567890]
		if len(sys.argv) > 3:
			seeds = [int(sys.argv[3])]

		for seed in seeds:
			np.random.seed(seed)
			random.seed(seed)
			train_data, train_label, test_data, test_label = data_hand.split()

			my_classifier = classifier.Classifier(train_data, train_label, test_data, test_label)
			my_mask = mask.Mask(len(train_data[0]))
			my_mask.randomize()

			t_start = time.time()
			for _ in range(100):
				score = my_classifier.new_score(my_mask,2)
			t_end = time.time()
			print("Score: " +  str(score))
			print("Tiempo: " + str(t_end - t_start))
			t_start = time.time()
			for _ in range(100):
				score = my_classifier.new_score(my_mask,4)
			t_end = time.time()
			print("Score: " +  str(score))
			print("Tiempo: " + str(t_end - t_start))
			t_start = time.time()
			for _ in range(100):
				score = my_classifier.score_train(my_mask)
			t_end = time.time()
			print("Score: " +  str(score))
			print("Tiempo: " + str(t_end - t_start))
			print(my_classifier.score_train(my_mask))

			my_heuristic.set_classifier(my_classifier)

			start_time = time.time()
			score = my_heuristic.run(my_mask,5000)
			end_time = time.time()

			reduction = 0

			for value in my_mask.values:
				if value:
					reduction += 1

			reduction = 100 * (len(my_mask.values)-reduction)/len(my_mask.values)
			out_score = my_classifier.score_test(my_mask) * 100

			print("Mascara obtenida")
			print(my_mask.values)
			print("Reduccion: " + str(reduction))
			print("Score de entrenamiento: " + str(score))
			print("Score de test: " + str(out_score))
			print("Tiempo transcurrido: " + str(end_time - start_time))

			my_classifier = classifier.Classifier(test_data, test_label,train_data, train_label)
			my_mask = mask.Mask(len(train_data[0]))
			my_mask.randomize()

			my_heuristic.set_classifier(my_classifier)

			start_time = time.time()
			score = my_heuristic.run(my_mask,5000)
			end_time = time.time()

			reduction = 0

			for value in my_mask.values:
				if value:
					reduction += 1

			reduction = 100 * (len(my_mask.values)-reduction)/len(my_mask.values)
			out_score = my_classifier.score_test(my_mask) * 100

			print("Mascara obtenida")
			print(my_mask.values)
			print("Reduccion: " + str(reduction))
			print("Score de entrenamiento: " + str(score))
			print("Score de test: " + str(out_score))
			print("Tiempo transcurrido: " + str(end_time - start_time))

	else:
		print("Número de parámetros incorrectos.")

if __name__ == "__main__":
	main()
