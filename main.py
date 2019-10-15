from network import *
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
			data = data.reshape((28,28))
			dataset.append(data)
			label = np.zeros((10,1))
			label[int(row[786])][0] = 1
			labels.append(label)
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

	myNet = Network(dataset, labels)
	print("outShape", myNet.output.shape)
	myNet.addLayer(Dense(28, 28, 28, 28))
	myNet.addLayer(Dense(28, 28, 10, 1))
	myNet.addLayer(Dense(10, 1, 10, 1))


	myNet.feedForward(myNet.dataset[0])
	#print(myNet.output)
	#print(myNet.output.shape)

	#for layer in myNet.layers:
		#print(layer.weights.shape)
		
if __name__== "__main__":
	main()

