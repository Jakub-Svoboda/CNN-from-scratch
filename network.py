import numpy as np
import activations


class Dense():
	
	def __init__(self, inputWidth, inputHeight, neuronsWidth, neuronsHeight):
		self.weightsWidth = inputWidth
		self.weightHeight = inputHeight
		self.neuronsWidth = neuronsWidth
		self.neuronsHeight = neuronsHeight
		self.layerOutput = None
		self.weights = np.random.rand(inputWidth*inputHeight, neuronsWidth*neuronsHeight)
		self.bias = np.random.rand(neuronsWidth*neuronsHeight)
		self.error = None
		self.delta = None	
		self.layerOutput = None	

	def forward(self, inputs):
		#print("Forward pass of layer", self.weightsWidth, self.weightHeight)
		self.layerOutput = activations.sigmoid(np.dot(inputs, self.weights) + self.bias)    #sig(x * w) +b
		return self.layerOutput


class Network():
	
	def __init__(self):
		self.layers = []

	def addLayer(self, layer):
		self.layers.append(layer)

	def feedForward(self, dataset, labels, index):
		inp = dataset[index]
		for layer in self.layers: 
			inp = layer.forward(inp)	#TODO it would be better not to pass inp here, but getting it inside the layer from the previous layer directly
		self.output = inp	

	def backProp(self, dataset, labels, index):
		for idx in reversed(range(len(self.layers))):
			layer = self.layers[idx]
			if layer == self.layers[-1]:   #if last layer
				#print("Backwards pass:", labels[index].shape, layer.layerOutput.shape)
				layer.error = labels[index] - self.output
				layer.delta = layer.error * activations.sigmoidDerivative(layer.layerOutput)
			else:
				nextLayer = self.layers[idx+1]
				#print("Backwards pass: NL.weights:", nextLayer.weights.shape, "NL.delta:", nextLayer.delta.shape)
				layer.error = np.dot(nextLayer.weights, nextLayer.delta)	
				layer.delta = layer.error * activations.sigmoidDerivative(layer.layerOutput)
		
	def updateWeights(self, dataset, labels, index, lr):	
		ldt = 0
		for idx, layer in enumerate(self.layers): #For each layer, update its weights
			if layer == self.layers[0]: 	#if last layer
				inputs = np.atleast_2d(dataset[index]) #the input is the image
			else:						
				inputs = np.atleast_2d(self.layers[idx-1].layerOutput) #the input is the previous layer's output	
			layer.weights += layer.delta * inputs.T * lr
			print(layer.delta)
			ldt += np.sum(layer.delta)
		print("ldt", ldt)	

	def fit(self, dataset, labels, lr, epochs):
		for epochNum in range(epochs):
			for stepNum in range(len(dataset)): 
				self.feedForward(dataset, labels, stepNum)
				self.backProp(dataset, labels, stepNum)
				self.updateWeights(dataset, labels, stepNum, lr)
				if stepNum % 10 == 0:
					print("Epoch:", epochNum+1, " Step", stepNum, "of", len(dataset))


		
		














	
		
