import numpy as np
import activations

class Neuron():
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.weights = np.random.rand(self.width, self.height)

	def forward(self, inputs):
		#print("forward pass of neuron", inputs.shape, self.weights.shape)
		#res = np.sum(np.outer(inputs, self.weights))	
		res = np.dot(inputs.flatten(), self.weights.flatten())
		return res				

class Dense():
	
	def __init__(self, width, height, outWidth, outHeight):
		self.width = width
		self.height = height
		self.neurons = [[Neuron(self.width, self.height) for j in range(outWidth)] for i in range(outHeight)]

	def forward(self, inputs):
		#print(inputs.shape, self.neurons.shape)
		print("Forward pass of layer")
		layerOutput = np.zeros(((self.outWidth, self.outHeight)))
		for idx, row in enumerate(layerOutput):
			for idx2, col in enumerate(row): #for each neuron
				res = self.neurons[idx][idx2].forward(inputs)
				#print(res)							

				layerOutput[idx][idx2] = activations.sigmoid(res)
		#layerOutput =  activations.sigmoid(np.dot(inputs, self.weights)) #dot product = sum of multiplications
		print(layerOutput.shape)
		print("----------")
		return layerOutput


class Network():
	
	def __init__(self, dataset, labels):
		self.layers = []
		self.dataset = dataset
		self.labels = labels
		self.output = np.zeros(labels[0].shape)
		
	def addLayer(self, layer):
		self.layers.append(layer)

	def feedForward(self, inp):
		inp = inp
		for layer in self.layers[:-1]:
			inp = layer.forward(inp)
		self.output = inp	
		print(self.output)

	'''def backprop(self):
		# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
		d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
		d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

		# update the weights with the derivative (slope) of the loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2
	'''
		
