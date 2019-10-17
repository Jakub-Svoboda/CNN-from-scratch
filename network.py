import numpy as np
import activations
from sklearn.model_selection import train_test_split


class Dense():
	def __init__(self, inputWidth, inputHeight, neuronsWidth, neuronsHeight):
		self.weightsWidth = inputWidth
		self.weightHeight = inputHeight
		self.neuronsWidth = neuronsWidth
		self.neuronsHeight = neuronsHeight
		self.layerOutput = None
		self.weights = np.random.rand(inputWidth*inputHeight, neuronsWidth*neuronsHeight)*0.001
		self.bias = np.zeros(neuronsWidth*neuronsHeight) #TODO ZERO OR rand?
		self.error = None
		self.delta = None	
		self.layerOutput = None	

	def forward(self, inputs):
		self.layerOutput = activations.sigmoid(np.dot(inputs, self.weights) + self.bias)    #sig(x * w) +b
		return self.layerOutput


class Network():
	def __init__(self):
		self.layers = []

	def addLayer(self, layer):
		self.layers.append(layer)

	def getValDataset(self, dataset, labels):
		valDataset = []
		valLabels = []
		dataset, valDataset, labels, valLabels = train_test_split(dataset, labels, test_size=0.10, random_state=42)
		return dataset, labels, valDataset, valLabels

	def feedForward(self, dataset, index=None):
		if index != None:
			inp = dataset[index]
		else:
			inp = dataset	
		for layer in self.layers: 
			inp = layer.forward(inp)	#TODO it would be better not to pass inp here, but getting it inside the layer from the previous layer directly
		self.output = inp	
		return inp
	def backProp(self, dataset, labels, index):
		for idx in reversed(range(len(self.layers))):
			layer = self.layers[idx]
			if layer == self.layers[-1]:   #if last layer
				layer.error = labels[index] - self.output
				layer.delta = layer.error * activations.sigmoid(layer.layerOutput)
			else:
				nextLayer = self.layers[idx+1]
				layer.error = np.dot(nextLayer.weights, nextLayer.delta)	
				layer.delta = layer.error * activations.sigmoidDerivative(layer.layerOutput)
		
	def updateWeights(self, dataset, labels, index, lr):	
		for idx, layer in enumerate(self.layers): #For each layer, update its weights
			if layer == self.layers[0]: 			#if last layer
				inputs = np.atleast_2d(dataset[index]) #the input is the image
			else:						
				inputs = np.atleast_2d(self.layers[idx-1].layerOutput) #the input is the previous layer's output	
			layer.weights += layer.delta * inputs.T * lr

	def predict(self, dataset, index):
		return self.feedForward(dataset, index)

	def evaluate(self, dataset, labels):
		outputs = []
		gt = []
		for sample in labels:
			gt.append(np.argmax(sample))
		for idx, _ in enumerate(dataset):
			#print(self.predict(dataset, idx))
			outputs.append(np.argmax(self.predict(dataset, idx)))
			print("GT:", gt[idx], "predicted:", outputs[idx], self.predict(dataset, idx))

	def fit(self, dataset, labels, lr, epochs):
		dataset, labels, valDataset, valLabels = self.getValDataset(dataset, labels)
		mse = []
		for epochNum in range(epochs):
			for stepNum in range(len(dataset)): 
				self.feedForward(dataset, stepNum)
				self.backProp(dataset, labels, stepNum)
				self.updateWeights(dataset, labels, stepNum, lr)

				se = np.square(labels[stepNum] - self.output)
				mse.append(se)
				if stepNum % 100 == 0:
					mse = np.mean(mse)
					print("Epoch:", epochNum+1, "MSE:", float(mse), " Step", stepNum, "of", len(dataset))
					mse = []
			self.evaluate(valDataset, valLabels)


		
		














	
		
