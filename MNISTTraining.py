from nnNumpy import *; # neural network
from time import time;

'''
from keras.datasets import mnist;
(x_train, y_train), (x_test, y_test) = mnist.load_data();

# adjusting the trainig data (DON'T FRICKING FORGET TO NORMALIZE)

normalization = 1 / 255;

trainingData = [];
for i in range(len(x_train)):
	trainingData.append([x_train[i].ravel() * normalization, [int(j == y_train[i]) for j in range(10)]]);

# creating the testing data
testingData = [];
for i in range(len(x_test)):
	testingData.append([x_test[i].ravel() * normalization, [int(j == y_test[i]) for j in range(10)]]);


print("Done loading!");
'''

def testNN(testingData, nn):
	correct = 0;
	for i in range(len(testingData)):
		x = testingData[i][0];
		y = testingData[i][1];

		networkY = nn.feedForward(x);

		correct += np.argmax(y) == np.argmax(networkY);

	#	if ((i + 1) % (len(testingData) / 10) == 0):
	#		print(f"{correct}/{i + 1}");

	print(f"The network got {correct}/{len(testingData)} correct. That is {(correct / len(testingData) * 100):.2f}% of correct answers.")

	return correct / len(testingData);





def trainMultiple(trainingData, nn):

	start = time();

	nn.SGD(trainingData, 1, 8, 2);

	print(f"Multiple done, took {(time() - start):.2f} seconds.");

def trainSingle(trainingData, nn):
	
	start = time();

	iterations = int(len(trainingData));

	for i in range(iterations):
		# data = choice(trainingData);
		data = trainingData[i];

		nn.train(
			rotateAndTranslateMat(data[0] + np.random.rand(len(data[0])) * 0.6 - 0.3, 28, 28, np.random.rand() - 0.5, int(np.random.rand() * 10 - 5), int(np.random.rand() * 10 - 5)), 
			data[1], 0.3, 0.05, iterations
		);

	#	if (i % int(iterations / 10) == 0):
	#		print(str(int(i / (iterations / 10) + 1)), "training sets");
	
	
	print(f"Single done, took {(time() - start):.2f} seconds.");


network = Network(2, 1, 1, 1, [activationSigmoid]);
# network = Network.loadFromFile("aug-96");


trainingData = [
	[[0, 0], [0]],
	[[1, 0], [1]],
	[[0, 1], [1]],
	[[1, 1], [0]]
];

network.SGD(trainingData, 100, 4, 3);









'''
x = input("Manda: ");
while x != "":

	x = list(map(int, x.split(",")));

	y = network.feedForward(x);
	print(y);
	print(f"Eu acho que é o número {np.argmax(y)}");


	x = input("Manda: ");
'''

'''
i = 0;
while True:
	try:
		np.random.shuffle(trainingData);
		trainMultiple(trainingData, network);
	except KeyboardInterrupt:
		print("Training stoped.");

	try:
		print(f"Epoch {i + 1} completed.");

		print("\nCasos de teste: ");
		testNN(testingData, network);

	#	print("\nCasos de treino: ");
	#	testNN(trainingData[:10000], network);
	#	print("\n\n");
	except KeyboardInterrupt:
		print("Testing stoped.");

		break;

	i += 1;


if (input("Quer salvar? (S|N)").strip().upper() == "S"):
	nome = input("Digite o nome do arquivo em que quer salvar: ");
	network.save(nome);
'''