import csv
import sys

def main():
	with open("AGG.csv",'a',newline='') as csvfile, open("AGG-wdbc.csv") as wdbc,open("AGG-libras.csv") as libras, open("AGG-arrhythmia.csv") as arr:
		spamwriter = csv.writer(csvfile, delimiter=' ')
		for d0,d1,d2 in zip(wdbc, libras, arr):
			spamwriter.writerow([d0.rstrip('\n')+' '+d1.rstrip('\n')+' '+d2])

if __name__ == "__main__":
	main()
