from helper import *;
from linearAlgebra import *;

import numpy as np;
from json import dump, load;

# activation functions

# commonly used for models where we have to predict the probability as an output.
def sigmoid(x):
	return 1 / (1 + np.exp(-x));
def sigmoidPrime(x):
	s = sigmoid(x);
	return s * (1 - s);



def binaryStep(x):
	return x >= 0;
def binaryStepPrime(x):
	return 0;



def linear(x):
	return x;
def linearPrime(x):
	return 1;



# usually used in hidden layers of a neural network
def tanh(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def tanhPrime(x):
	b = np.exp(x) + np.exp(-x);
	return 4 / (b * b);



# bad: the negative side of the graph makes the gradient value zero (bad backpropagation)
def ReLU(x):
	return x * (x > 0);
def ReLUPrime(x):
	return binaryStep(x);



def leakyReLU(x, a = 0.1):
	return np.maximum(a * x, x);
def leakyReLUPrime(x, a = 0.1):
	return a + (1 - a) * (x > 0);



# becomes smooth slowly until its output equal to -α
# avoids dead ReLU problem
def ELU(x, a = 1.6732632423):
	return x * (x >= 0) + a * (np.exp(x) - 1) * (x < 0);
def ELUPrime(x, a = 1.6732632423):
	if (x < 0):
		return ELU(x, a) + a;

	return 1;



def swish(x):
	return x * sigmoid(x);
def swishPrime(x):
	return sigmoid(x) + x * sigmoidPrime(x);



# good for computer vision, natural language processing, and speech recognition
def GELU(x):
	sqrt2OverPI = 0.7978845608;
	x3 = x * x * x;
	return 0.5 * x * (1 + tanh(sqrt2OverPI * (x + 0.044715 * x3)));
def GELUPrime(x): # I calculated this myself and didn't pay much attention so this might be wrong
	sqrt2OverPI = 0.7978845608;
	x2 = x * x;
	x3 = x2 * x;
	f = sqrt2OverPI * (x + 0.044715 * x3);
	g = 1 + tanh(f);

	return 0.5 * (g + x * (tanhPrime(f) * sqrt2OverPI * (1 + 0.134145 * x2)));



# network converges faster (in comparisson to ReLU)
def SELU(x, l = 1.0507009873):
	return l * ELU(x);
def SELUPrime(x, l = 1.0507009873):
	return ELUPrime(x);



class ActivationFunction:
	def __init__(self, function = sigmoid, derivative = sigmoidPrime):
		self.function = function;
		self.derivative = derivative;

activationSigmoid = ActivationFunction(sigmoid, sigmoidPrime);
activationTanh = ActivationFunction(tanh, tanhPrime);
activationLinear = ActivationFunction(linear, linearPrime);
activationLeakyReLU = ActivationFunction(leakyReLU, leakyReLUPrime);






# cost function derivative to the mean squared error
def costDerivative(x, y):
	return y - x;






class Network:
	def __init__(self, inputSize, outputSize, hiddenLayers, hiddenLayersSize, activationFunctions = []):

		layersSize = [inputSize];
		for i in range(hiddenLayers):
			layersSize.append(hiddenLayersSize);
		layersSize.append(outputSize);


		for i in range(len(activationFunctions)):
			activationFunctions[i] = activationFunctions[i] or activationSigmoid;

		for i in range(len(layersSize) - len(activationFunctions) - 1):
			activationFunctions.append(activationSigmoid);


		self.layers = [];
		for i in range(len(layersSize) - 1):
			self.layers.append(Layer(layersSize[i], layersSize[i + 1], activationFunctions[i]));



	def feedForward(self, givenInput):
		output = self.layers[0].feedForward(givenInput);
		for i in range(1, len(self.layers)):
			output = self.layers[i].feedForward(output);

		return output;

	def mutate(amount = 1):
		for i in range(len(self.layers)):
			layer = self.layers[i];

			for j in range(len(layer.bias)):
				layer.bias[j] = lerp(layer.bias[j], random() * 2 - 1, amount);

			for j in range(len(layer.weights)):
				for k in range(len(layer.weights[j])):
					layer.weights[j][k] = lerp(layer.weights[j][k], random() * 2 - 1, amount);



	def SGD(self, trainingData, epochs, miniBatchSize, learningRate):

		for i in range(epochs):
			np.random.shuffle(trainingData);

			miniBatches = [];
			for j in range(0, len(trainingData), miniBatchSize):
				miniBatches.append(transpose(trainingData[j:j + miniBatchSize]));

			for miniBatch in miniBatches:
				self.updateMiniBatch(miniBatch, learningRate);
			
			print(f"Epoch {i} complete");


	def updateMiniBatch(self, miniBatch, learningRate):

		deltaNablaB, deltaNablaW = self.backpropagate(np.array(miniBatch[0]), np.array(miniBatch[1]));

		learningRate /= len(miniBatch);

		# update the weights and biases
		for i in range(len(self.layers)):

			self.layers[i].bias += sum(deltaNablaB[0][i]) * learningRate;
			self.layers[i].weights += sum(deltaNablaW[0][i]) * learningRate;





	def train(self, givenInput, target, learningRate = 1, regularization = 5, dataLength = 1):

		deltaNablaB, deltaNablaW = self.backpropagate(givenInput, target);

		for i in range(len(self.layers)):

			# L2 regularization
		#	self.layers[i].weights *= (1 - learningRate * (regularization / dataLength));

			self.layers[i].bias += deltaNablaB[i] * learningRate;
			self.layers[i].weights += deltaNablaW[i] * learningRate;


	
	def backpropagate(self, x = [[]], y = [[]]):
		x = x.transpose(); y = y.transpose();
		
		nablaB = [np.zeros(np.shape(layer.bias)) for layer in self.layers];
		nablaW = [np.zeros(np.shape(layer.weights)) for layer in self.layers];

		activations = [x]; # activation of each neuron per layer (values aplyed to the activation function)
		zs = []; # values of each neuron per layer

		for i in range(len(self.layers)):
			b = self.layers[i].bias;
			w = self.layers[i].weights;

			print(0)
			print(w, "W")
			print(activations[i], "A")

			z = np.dot(w, activations[i]) + b;
			activation = self.layers[i].activation.function(z);

			zs.append(z);
			activations.append(activation);


		delta = costDerivative(activations[-1], y) * self.layers[-1].activation.derivative(zs[-1]);

		nablaB[-1] = delta;
		nablaW[-1] = np.outer(delta, activations[-2]);

		# backpropagate the error
		for l in range(2, len(self.layers) + 1):

			delta = np.dot(self.layers[-l + 1].weights.transpose(), delta) * self.layers[-l].activation.derivative(zs[-l]);

			nablaB[-l] = delta;
			nablaW[-l] = np.outer(delta, activations[-l - 1]);

		return nablaB, nablaW;
	


	def save(self, fileName = "neural network"):
		if (not fileName):
			fileName = "neural network";

		# creating an array of biases and weights
		bias = [];
		weights = [];
		for i in range(len(self.layers)):
			bias.append(self.layers[i].bias.tolist());
			weights.append(self.layers[i].weights.tolist());

		data = {
			"w": weights,
			"b": bias
		};

		# saving them to a file (json)
		with open(fileName + ".json", "w") as f:
			dump(data, f);
			f.close();

		# saving them to a file (numpy)
	#	np.save(fileName + ".npy", np.array([bias, weights], dtype = np.object));


	@staticmethod
	def loadFromFile(fileName = "neural network"):
		if (not fileName):
			fileName = "neural network";

		with open(fileName + ".json", "r") as f:
			data = load(f);
			f.close();

		weights = [np.array(w) for w in data["w"]];
		bias = [np.array(b) for b in data["b"]];

		inputSize = len(weights[0][0]);
		hiddenLayers = len(bias) - 1;
		hiddenLayersSize = len(weights[0]);
		outputSize = len(bias[hiddenLayers]);

		network = Network(inputSize, outputSize, hiddenLayers, hiddenLayersSize);

		for i in range(len(network.layers)):

			for j in range(len(network.layers[i].bias)):
				network.layers[i].bias[j] = bias[i][j];

				for k in range(len(network.layers[i].weights[j])):
					network.layers[i].weights[j][k] = weights[i][j][k];

		print(f"Loaded a neural network with {hiddenLayersSize} neurons in the hidden layers");

		return network;








class Layer:
	def __init__(self, inputSize, outputSize, activation = activationSigmoid):
		self.input = [0.0 for i in range(inputSize)];
		self.output = [0.0 for i in range(outputSize)];
		self.bias = [0.0 for i in range(outputSize)];
		self.activation = activation;

		self.weights = [];
		for i in range(outputSize):
			self.weights.append([0.0 for j in range(inputSize)]);

		self.input = np.array(self.input);
		self.output = np.array(self.output);
		self.bias = np.array(self.bias);
		self.weights = np.array(self.weights);

		self.randomize();

	def randomize(self):

		# uses Xavier initialization

		for i in range(len(self.weights)):
			for j in range(len(self.weights[i])):
				self.weights[i][j] = np.random.rand() / np.sqrt(len(self.input));
			#	self.weights[i][j] = np.random.rand() * 2 - 1;

		for i in range(len(self.bias)):
			self.bias[i] = np.random.rand() * 2 - 1;


	def feedForward(self, givenInput):
		return self.activation.function(np.dot(self.weights, givenInput) + self.bias);