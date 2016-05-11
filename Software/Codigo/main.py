import time
import heuristic, mask, data_handler, classifier
import numpy as np
import random
import csv
import sys
from knnGPU.knnLooGPU import knnLooGPU

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
		elif sys.argv[2] == "BMB":
			my_heuristic = heuristic.BMBHeuristic(None)
		elif sys.argv[2] == "GRASP":
			my_heuristic = heuristic.GRASPHeuristic(None)
		elif sys.argv[2] == "ILS":
			my_heuristic = heuristic.ILSHeuristic(None)
		elif sys.argv[2] == "AGG":
			my_heuristic = heuristic.AGGHeuristic(None)
		elif sys.argv[2] == "AGE":
			my_heuristic = heuristic.AGGHeuristic(None)
		else:
			print("Opcion de heuristica incorrecta")
			return

		file_name = "./Results/"+sys.argv[2]+"-"+sys.argv[1]+".csv"

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

			my_heuristic.set_classifier(my_classifier)

			start_time = time.time()
			score = my_heuristic.run(my_mask,5000)
			end_time = time.time()

			reduction = 0

			for value in my_mask.values:
				if value:
					reduction += 1

			reduction = 100 * (len(my_mask.values)-reduction)/len(my_mask.values)
			out_score = my_classifier.score_test(my_mask)

			score = round(score,2)
			out_score = round(out_score,2)
			reduction = round(reduction,2)
			total_time = round(end_time-start_time,2)
			print("Mascara obtenida")
			print(my_mask.values)
			print("Reduccion: " + str(reduction))
			print("Score de entrenamiento: " + str(score))
			print("Score de test: " + str(out_score))
			print("Tiempo transcurrido: " + str(total_time))

			with open(file_name,'a',newline='') as csvfile:
				spamwriter = csv.writer(csvfile, delimiter=' ',
				                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
				spamwriter.writerow([score,out_score,reduction,total_time])

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
			out_score = my_classifier.score_test(my_mask)

			score = round(score,2)
			out_score = round(out_score,2)
			reduction = round(reduction,2)
			total_time = round(end_time-start_time,2)
			print("Mascara obtenida")
			print(my_mask.values)
			print("Reduccion: " + str(reduction))
			print("Score de entrenamiento: " + str(score))
			print("Score de test: " + str(out_score))
			print("Tiempo transcurrido: " + str(total_time))

			with open(file_name,'a',newline='') as csvfile:
				spamwriter = csv.writer(csvfile, delimiter=' ',
				                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
				spamwriter.writerow([score,out_score,reduction,total_time])

	else:
		print("Número de parámetros incorrectos.")

if __name__ == "__main__":
	main()
