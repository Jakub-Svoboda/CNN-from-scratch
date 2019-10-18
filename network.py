import numpy as np
import activations
from sklearn.model_selection import train_test_split

class Conv3x3:
  # A Convolution layer using 3x3 filters.

	def __init__(self, numFilters):
		self.layerOutput = None
		self.numFilters = numFilters
		self.filters = np.random.randn(self.numFilters, 3, 3) / (3*3) # div for xavier initialization, these are the weights for the layers

	def feedForward(self, inputs):	
		height, width = inputs.shape
		self.layerOutput = np.zeros((height - 2, width -2, self.numFilters))

		for h in range(height - 2):				
			for w in range(width - 2): 	#for each filter/neuron
				subImage = inputs[h:(h + 3), w:(w + 3)]	#3x3 matrix from the input image
				mult = subImage * self.filters
				self.layerOutput[h,w] = np.sum(mult, axis=(1, 2))
		return self.layerOutput 


class MaxPool2x2:
	def __init__(self):
		self.layerOutput = None

	def feedForward(self, inputs):	
		height, width, filters = inputs.shape				
		self.layerOutput = np.zeros((height//2, width//2, filters))	
		for h in range(height//2):				
			for w in range(width//2): 	#for each 2 pixels/inputs
				self.layerOutput[h, w] = np.max(inputs[(h*2):(h*2+2),(2*2):(2*2+2)], axis = (0,1))		#we assume even number of pixels
		return self.layerOutput


class Softmax:
  # A standard fully-connected layer with softmax activation.

	def __init__(self, inputSize, neurons):
		self.inputSize = inputSize
		self.neurons = neurons
		self.weights = np.random.randn(inputSize, neurons) / inputSize
		self.biases = np.zeros(neurons)

	def feedForward(self, inputs):
		inputs = inputs.flatten()
		self.mult = np.dot(inputs, self.weights) + self.biases # y = W*X+b
		self.layerOutput = self.mult
		self.layerOutput = np.exp(self.layerOutput)						#softmax
		self.layerOutput = self.layerOutput / np.sum(self.layerOutput, axis=0)	#the sum of probabilities is now ~= 1
		return self.layerOutput

	def backProp(self, gradient):
		for idx, g in enumerate(gradient):
			if g == 0:
				continue
			tExp = np.exp(self.mult)			#exponential of the last weight*input +b
			s = np.sum(tExp)
			dOutDt = -tExp[idx] * tExp/(s**2)
			dOutDt[idx] = tExp[idx] * (s - tExp[idx])/(s**2)




class Network():
	def __init__(self):
		self.layers = []
		self.output = None

	def addLayer(self, layer):
		self.layers.append(layer)

	def initGradient(self, label): #assumes softmax is the last layer
		gradient = np.zeros(label.size)	
		labelDigit = np.argmax(label)
		gradient[labelDigit] = -(1/self.output[labelDigit])
		return gradient

	def fit(self, dataset, labels, lr, epochs):
		loss = 0
		correct = 0	

		for epochNum in range(epochs):
			for stepNum, img in enumerate(dataset):			#for each datum
				_, l, accuracy = self.feedForward(img, labels[stepNum])   		#feed forward
				gradient = self.initGradient(labels[stepNum])	
			


				loss += l
				correct += accuracy
				if (stepNum+1) % 100 == 0:
					print('Epoch: %d Step %d \tAverage Loss %.3f \tAccuracy: %d%%' % (epochNum+1, stepNum + 1, loss / 100, correct))	
					loss = 0
					correct = 0
			print("----- Epoch", epochNum+1, "finished -----")		

	def feedForward(self, img, label):
		for idx, layer in enumerate(self.layers):	#feed forward the network (each layer)
			if idx == 0:						#send image to the first layer
				self.output = layer.feedForward(img)
			else:								#send previous layer's output 
				self.output = layer.feedForward(self.layers[idx-1].layerOutput)	
		labelDigit = np.argmax(label)
		loss = -np.log(self.output[labelDigit]) #cross-entrophy loss calculation
		if labelDigit == np.argmax(self.output):	#if correct prediction
			accuracy = 1
		else:
			accuracy = 0

		return self.output, loss, accuracy
		

		
		














	
		
