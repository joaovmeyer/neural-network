from random import uniform;
from math import sin, cos;

# helper variables
E = 2.7182818284;
PI = 3.1415926535;

# helper functions
def random():
	return uniform(0, 1);

def choice(array):
	return array[int(random() * len(array))];

def shuffle(array):
	i = len(array);
	while (i):
		rand = int(random() * i); # random() * i is always positive, so int is equal to floor
		i -= 1;

		temporary = array[i];
		array[i] = array[rand];
		array[rand] = temporary;

def lerp(a, b, t):
	return a + (b - a) * t;

def abs(x):
	return x - 2 * x * (x < 0);

def exp(x):
	return E ** x;


def translateMat(mat, w, h, deltaX, deltaY): # mat is a 1D array representing a 2D image, so the first 2 elements are the first row etc.
	translated = [0 for i in range(len(mat))];

	for i in range(h):
		for j in range(w):
			index = w * i + j;

			if (not mat[index]):
				continue;

			# aply the translation
			newX = j + int(deltaX);
			newY = i + int(deltaY);

			if (newX < 0 or newX >= w or newY < 0 or newY >= h):
				continue;

			newIndex = w * newY + newX;
			translated[newIndex] = mat[index];

	return translated;

def rotateMat(mat, w, h, angle):
	rotated = [0 for i in range(len(mat))];
	
	cosTheta = cos(angle);
	sinTheta = sin(angle);
	
	for i in range(h):
		for j in range(w):
			index = w * i + j;
			
			if (not mat[index]):
				continue;
			
			x = j - ((w - 1) / 2);
			y = -i + ((h - 1) / 2);
			
			# aply the rotation
			newX = int(x * cosTheta - y * sinTheta + ((w - 1) / 2));
			newY = -int(x * sinTheta + y * cosTheta - ((h - 1) / 2));
			
			if (newX < 0 or newX >= w or newY < 0 or newY >= h):
				continue;
			
			newIndex = w * newY + newX;
			rotated[newIndex] = mat[index];
			
	return rotated;

def rotateAndTranslateMat(mat, w, h, angle, deltaX, deltaY):
	rotatedAndTranslated = [0 for i in range(w * h)];
	
	cosTheta = cos(angle);
	sinTheta = sin(angle);
	
	for i in range(h):
		for j in range(w):
			index = w * i + j;
			
			if (not mat[index]):
				continue;
			
			x = j - ((w - 1) / 2);
			y = -i + ((h - 1) / 2);
			
			# aply the rotation
			newX = int(x * cosTheta - y * sinTheta + ((w - 1) / 2) + deltaX);
			newY = -int(x * sinTheta + y * cosTheta - ((h - 1) / 2) - deltaY);
			
			if (newX < 0 or newX >= w or newY < 0 or newY >= h):
				continue;
			
			newIndex = w * newY + newX;
			rotatedAndTranslated[newIndex] = mat[index];
			
	return rotatedAndTranslated;
