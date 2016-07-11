import csv
import sys

def main():
	name = "AM3"

	with open(name+".csv",'a',newline='') as csvfile, open(name+"-wdbc.csv") as wdbc,open(name+"-libras.csv") as libras, open(name+"-arrhythmia.csv") as arr:
		spamwriter = csv.writer(csvfile, delimiter=' ')
		for d0,d1,d2 in zip(wdbc, libras, arr):
			spamwriter.writerow([d0.rstrip('\n')+' '+d1.rstrip('\n')+' '+d2])

if __name__ == "__main__":
	main()
