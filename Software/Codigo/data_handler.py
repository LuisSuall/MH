from scipy.io import arff
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

class DataHandler:

	def __init__(self):
		self.__data_values = None
		self.__data_label = None

	def load_libras(self):

		l = []
		m = MinMaxScaler()
		raw_data, metadata = arff.loadarff('../Datos/movement_libras.arff')

		for data in raw_data:
			l.append(list(data))

		self.__data_values = np.array([l[0][0:90]], dtype = np.float32)
		self.__data_label = np.array(l[0][90], dtype = np.int32)

		for i in range(1,len(l)):
			self.__data_values = np.append(self.__data_values,np.array([l[i][0:90]], dtype = np.float32), axis = 0)
			self.__data_label = np.append(self.__data_label,np.array(l[i][90], dtype=np.int32))

		print("Los datos")
		print(self.__data_values[0])
		print("Los labels")
		print(self.__data_label)

		self.__data_values = m.fit_transform(self.__data_values)
		print("Tras el fit, queda así")
		print(self.__data_values[0])
		print(self.__data_values[0].dtype)

	def load_arrhythmia(self):

		l = []
		m = MinMaxScaler()
		raw_data, metadata = arff.loadarff('../Datos/arrhythmia.arff')

		for data in raw_data:
			l.append(list(data))

		self.__data_values = np.array([l[0][0:278]], dtype = np.float32)
		self.__data_label = np.array(l[0][278], dtype = np.int32)

		for i in range(1,len(l)):
			self.__data_values = np.append(self.__data_values,np.array([l[i][0:278]], dtype = np.float32), axis = 0)
			self.__data_label = np.append(self.__data_label,np.array(l[i][278], dtype=np.int32))

		print("Los datos")
		print(self.__data_values[0])
		print("Los labels")
		print(self.__data_label)

		self.__data_values = m.fit_transform(self.__data_values)

	def load_wdbc(self):

		l = []
		m = MinMaxScaler()
		raw_data, metadata = arff.loadarff('../Datos/wdbc.arff')

		for data in raw_data:
			l.append(list(data))

		self.__data_values = np.array([l[0][1:31]], dtype = np.float32)
		self.__data_label = np.array(l[0][0], dtype = np.int32) #TODO :usar code o decode

		for i in range(1,len(l)):
			self.__data_values = np.append(self.__data_values,np.array([l[i][1:31]], dtype = np.float32), axis = 0)
			self.__data_label = np.append(self.__data_label,np.array(l[i][0]), dtype = np.int32)

		self.__data_values = m.fit_transform(self.__data_values)

	def split(self):
		skf = StratifiedKFold(self.__data_label, 2, shuffle = True)

		for train, test in skf:
			return self.__data_values[train], self.__data_label[train],self.__data_values[test], self.__data_label[test]
