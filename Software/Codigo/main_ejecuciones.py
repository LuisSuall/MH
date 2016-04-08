import time
import heuristic, mask, data_handler, classifier
import numpy as np
import random
import csv

def main():
	data_hand = data_handler.DataHandler()
	data_hand.load_arrhythmia()

	seeds = [12345678,90123456,78901234,456789012,34567890]

	for seed in seeds:
		np.random.seed(seed)
		random.seed(seed)
		train_data, train_label, test_data, test_label = data_hand.split()

		my_classifier = classifier.Classifier(train_data, train_label, test_data, test_label)
		my_mask = mask.Mask(len(train_data[0]))
		my_mask.randomize()

		my_heuristic = heuristic.TABUHeuristic(my_classifier)

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

		with open('TABUarrhy.csv','a',newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=' ',
			                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow([score,out_score,reduction,end_time - start_time])

		my_classifier = classifier.Classifier(test_data, test_label,train_data, train_label)
		my_mask = mask.Mask(len(train_data[0]))
		my_mask.randomize()

		my_heuristic = heuristic.TABUHeuristic(my_classifier)

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

		with open('TABUarrhy.csv','a',newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=' ',
			                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow([score,out_score,reduction,end_time - start_time])


if __name__ == "__main__":
	main()
