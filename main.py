import network
import csv
import numpy as np
import activations
import argparse
import sys
from sklearn.model_selection import train_test_split

def splitDataset(dataset, labels, split):
	if split < 0:
		print("Cannot make split with negative value.")
		exit(1)
	dataset, testDataset, labels, testLabels = train_test_split(dataset, labels, test_size = split, random_state = 42)	
	return dataset, labels, testDataset, testLabels

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
			labels.append(label) 		
			lineCount += 1
	dataset = np.array(dataset)
	dataset = (dataset / 255) - 0.5 #normalize to (-0.5,0.5)
	labels = np.array(labels)	
	return dataset, labels	

def setArguments(args):
	parser = argparse.ArgumentParser(description = "Train and evaluation script for CNN.")
	parser.add_argument("--train", "-t", help = "The amount of training epochs should be used. If 0 or none, the network will not train.", default=0, type = int)
	parser.add_argument("--save", "-s", help = "Where to save the trained model.", default = "model.pckl", type = str)
	parser.add_argument("--load", "-l", help = "Model to load.", default = "model.pckl", type = str)
	args = parser.parse_args()
	return args

def main(args=None):
	np.random.seed(42)	

	if args is None:
		args = sys.argv[1:]
	args = setArguments(args)	

	print("Loading dataset...")
	dataset, labels = loadDataset()
	dataset, labels, testDataset, testLabels = splitDataset(dataset, labels, 0.1)
	print("Dataset loaded, dataset shape:", dataset.shape, ", labels shape:", labels.shape)
	print("Test dataset loaded, shape:", testDataset.shape, "test labels shape:", testLabels.shape)

	if args.train > 0: 				#create network and train
		myNet = network.Network()
		myNet.addLayer(network.Conv3x3(8))
		myNet.addLayer(network.MaxPool2x2())
		myNet.addLayer(network.Softmax(13*13*8, 10))
		myNet.fit(dataset, labels, 0.005, args.train)			#TODO right now the dataset size is limited
		network.saveModel(myNet)
	else:							#load model from disk
		myNet = network.loadModel(args.load)

	myNet.test(testDataset, testLabels)
	
if __name__== "__main__":
	main()

