import network
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
			label = label.reshape(10,)
			labels.append(label) 		#transpose for having (1,10) shape
			lineCount += 1
	dataset = np.array(dataset)
	dataset = (dataset - np.min(dataset))/np.ptp(dataset) + 0.5 #normalize to (-0.5,0.5)
	labels = np.array(labels)	
	return dataset, labels	

def main(args=None):
	random.seed(42)

	print("Loading dataset...")
	dataset, labels = loadDataset()
	print("Dataset loaded, dataset shape:", dataset.shape, ", labels shape:", labels.shape)

	myNet = network.Network()
	myNet.addLayer(network.Conv3x3(8))
	myNet.addLayer(network.MaxPool2x2())
	myNet.addLayer(network.Softmax(13*13*8, 10))


	myNet.fit(dataset[:1000], labels[:1000], 0.0003, 3)			#TODO right now the dataset size is limited
		
if __name__== "__main__":
	main()

