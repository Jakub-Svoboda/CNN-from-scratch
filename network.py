import numpy as np
import activations
from sklearn.model_selection import train_test_split
import pickle

def saveModel(net, path = "model.pckl"):
	try:
		myFile = open("model.pckl", 'wb')
		pickle.dump(net, myFile)
		myFile.close()
		print("Model saved.")
	except:
		print("Saving model failed.")	
		exit(1)

def loadModel(path = "model.pckl"):
	try:
		with open("model.pckl", 'rb') as myFile:
			net = pickle.load(myFile)
			print("Model loaded.")
			return net
	except:
		print("Loading model failed.")	
		exit(1)	


class Conv3x3:
	# A Convolution layer using 3x3 filters.
	def __init__(self, numFilters):
		self.layerOutput = None
		self.numFilters = numFilters
		self.filters = np.random.randn(self.numFilters, 3, 3) / (3*3) # div for xavier initialization, these are the weights for the layers

	def feedForward(self, inputs):	
		self.inputs = inputs
		height, width = inputs.shape
		self.layerOutput = np.zeros((height - 2, width -2, self.numFilters))

		for h in range(height - 2):				
			for w in range(width - 2): 	#for each filter/neuron
				subImage = inputs[h:(h + 3), w:(w + 3)]	#3x3 matrix from the input image
				mult = subImage * self.filters
				self.layerOutput[h,w] = np.sum(mult, axis=(1, 2))
		#self.mult = self.layerOutput		
		#self.layerOutput = activations.ReLU(self.layerOutput)		
		return self.layerOutput 

	def backProp(self, gradient):
		self.dWeights = np.zeros(self.filters.shape)	#delta weights is set with the shape of weights matrix
		height, width = self.inputs.shape
		for h in range(height - 2):
			for w in range(width - 2):
				subImage = self.inputs[h:(h + 3), w:(w + 3)]
				for f in range(self.numFilters):
					self.dWeights[f] += gradient[h, w, f] * subImage
		return self.dWeights

	def update(self, lr):
		self.filters -= self.dWeights *lr	


class MaxPool2x2:
	def __init__(self):
		self.layerOutput = None

	def feedForward(self, inputs):	
		self.inputs = inputs
		height, width, filters = inputs.shape				
		self.layerOutput = np.zeros((height//2, width//2, filters))	
		for h in range(height//2):				
			for w in range(width//2): 	#for each 2 pixels/inputs
				self.layerOutput[h, w] = np.max(inputs[(h*2):(h*2+2),(w*2):(w*2+2)], axis = (0,1))		#we assume even number of pixels	
		return self.layerOutput

	def backProp(self, gradient):
		dLdInput = np.zeros(self.inputs.shape)
		height, width, _ = self.inputs.shape
		height = height//2
		width = width//2
		for h in range(height):
			for w in range(width):		#for each output data point
				subImage = self.inputs[(h*2):(h*2+2),(w*2):(w*2+2)]
				h2, w2, f2 = subImage.shape
				m = np.max(subImage, axis=(0,1))

				for i in range(h2):
					for j in range(w2):
						for f in range(f2): #for each subimage(a 3x3 input data point)
							if subImage[i, j, f] == m[f]:		#for the maximal pixel
								dLdInput[h * 2 + i, w * 2 + j, f] = gradient[h, w, f]		
		return dLdInput

	def update(self, lr):
		pass	#max pooling layers has no weights to train


class Softmax:
  # A standard fully-connected layer with softmax activation.

	def __init__(self, inputSize, neurons):
		self.inputSize = inputSize
		self.neurons = neurons
		self.weights = np.random.randn(inputSize, neurons) / inputSize
		self.biases = np.zeros(neurons)

	def feedForward(self, inputs):
		self.inputShape = inputs.shape
		self.inputs = inputs.flatten()
		self.mult = np.dot(self.inputs, self.weights) + self.biases # y = W*X+b
		self.layerOutput = self.mult
		self.layerOutput = np.exp(self.layerOutput)						#softmax
		self.layerOutput = self.layerOutput / np.sum(self.layerOutput, axis=0)	#the sum of probabilities is now ~= 1
		return self.layerOutput

	def backProp(self, gradient):
		for idx, g in enumerate(gradient):
			if g == 0:			#if not the nonzero class, then continue
				continue
			tExp = np.exp(self.mult)			#exponential of the last (weight*input +b)

			s = np.sum(tExp)
			dOutdT = -tExp[idx] * tExp/(s**2)
			dOutdT[idx] = tExp[idx] * (s - tExp[idx])/(s**2)

			#Gradient of loss against (XW +b)
			self.dBiases = g * dOutdT

			#Calculate the gradient for weights, biases and inputs:
			self.dWeights = np.matmul(self.inputs[np.newaxis].T, self.dBiases[np.newaxis])   #delta of weights (matrix multiplication)								#delta if biases
			self.dInputs = np.matmul(self.weights, self.dBiases)							#delta if input	
			self.dInputs = self.dInputs.reshape(self.inputShape) #reverse the flattening
			return self.dInputs
	
	def update(self, lr):
		self.dWeights = self.dWeights * lr			#TODO <-waat?
		self.weights -= self.dWeights
		self.biases -= self.dBiases * lr
			

class Network():
	def __init__(self):
		self.layers = []
		self.output = None

	def addLayer(self, layer):
		self.layers.append(layer)

	def initGradient(self, label): #assumes softmax is the last layer
		gradient = np.zeros(label.size)				
		labelDigit = np.argmax(label)
		gradient[labelDigit] = -(1/self.output[labelDigit])	#deltaCost/deltaOut
		return gradient

	def fit(self, dataset, labels, lr, epochs):
		loss = 0
		correct = 0	

		for epochNum in range(epochs):
			print("----- Epoch", epochNum+1, "started -----")
			for stepNum, img in enumerate(dataset):			#for each datum
				_, l, accuracy = self.feedForward(img, labels[stepNum])   		#feed forward
				gradient = self.initGradient(labels[stepNum])			
				self.backProp(gradient)											#propagate backwards
				self.update(lr)

				loss += l
				correct += accuracy
				if (stepNum+1) % 100 == 0:
					print('Epoch: %d Step %d \tAverage Loss %.3f \tAccuracy: %d%%' % (epochNum+1, stepNum + 1, loss / 100, correct))	
					loss = 0
					correct = 0	

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
		
	def backProp(self, gradient):
		for idx, layer in enumerate(self.layers[::-1]): # backprop for each layer
			if idx == 0:				# if last layer
				dInputs = layer.backProp(gradient)
			else:
				dInputs = layer.backProp(dInputs)	
	
	def update(self, lr):
		for layer in self.layers:
			layer.update(lr)

	def test(self, dataset, labels):
		print("----- Testing started: -----")
		accuracy = 0
		for idx, datum in enumerate(dataset):
			_, _, a = self.feedForward(datum, labels[idx])
			accuracy += a
		print("Network predicted accurately in %.1f%% of cases." % ((accuracy/idx)*100))




		














	
		
