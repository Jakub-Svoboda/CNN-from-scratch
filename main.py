from network import Network, Dense
import csv
import numpy as np
import random
import activations

def loadDataset():
	with open("dataset/mnist-tk.inp") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=' ')
		lineCount = 0
		dataset = []
		labels = []
		for row in csv_reader:
			dataInt = []
			for idx, num in enumerate(row):
				if idx == 784:
					break
				dataInt.append(int(num))	
			data = np.array(dataInt)
			#data = data.reshape((28,28))
			dataset.append(data)
			label = np.zeros((10,1))
			label[int(row[786])][0] = 1
			label = label.reshape(10,)
			labels.append(label) 		#transpose for having (1,10) shape
			lineCount += 1
	dataset = np.array(dataset)
	dataset = (dataset - np.min(dataset))/np.ptp(dataset)
	labels = np.array(labels)	
	return dataset, labels	

def main(args=None):
	random.seed(42)

	print("Loading dataset...")
	dataset, labels = loadDataset()
	print("Dataset loaded, dataset shape:", dataset.shape, ", labels shape:", labels.shape)

	myNet = Network()
	myNet.addLayer(Dense(28, 28, 28, 28))
	myNet.addLayer(Dense(28, 28, 1, 10))

	myNet.fit(dataset, labels, 0.0003, 3)
		
if __name__== "__main__":
	main()

