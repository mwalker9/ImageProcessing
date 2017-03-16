import cv2
import numpy as np
import random
from itertools import product
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import svm
from os import listdir
from os.path import isfile, join
import math
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

def getDescriptors(img):
	sift = cv2.SIFT()
	kp, des = sift.detectAndCompute(img, None)
	return des
	
def getPrototypes(k, data):
	np.random.seed(5)
	est = KMeans(n_clusters=k)
	est.fit(data)
	return est
	
def train(X, y):
	clf = svm.LinearSVC()
	clf.fit(X, y)
	return clf
	
def predict(X, clf):
	return clf.predict(X)

def getPrototypeNumber(data, est):
	return est.predict(data)

def getAllImageDescriptors(directoryName):
	print("Loading: "+directoryName)
	images = []
	for filename in listdir(directoryName):
		str = join(directoryName,filename)
		img = cv2.imread(str)
		images.append(getDescriptors(img))
	return images
	
def run():
	imgTypes = []
	imgTypes.append(getAllImageDescriptors("101_ObjectCategories\car_side"))
	imgTypes.append(getAllImageDescriptors("101_ObjectCategories\\bonsai"))
	imgTypes.append(getAllImageDescriptors("101_ObjectCategories\\brain"))
	imgTypes.append(getAllImageDescriptors("101_ObjectCategories\\butterfly"))
	imgTypes.append(getAllImageDescriptors("101_ObjectCategories\chandelier"))
	percentageForTraining = .7
	k = 200
	neededNumberOfDescriptors = 100 * k
	numberOfDescriptorsPerType = math.ceil(neededNumberOfDescriptors / len(imgTypes))
	dsc = []
	for type in imgTypes:
		numberOfDescriptorsPerImage = math.ceil(numberOfDescriptorsPerType / len(type))
		for img in type:
			m = min(numberOfDescriptorsPerImage, len(img))
			n = len(img) 
			randomSet = np.random.permutation(n)[0:m]
			for l in randomSet:
				dsc.append(img[l])
	est = getPrototypes(k, dsc)
	histogramedImgTypes = []
	print("centroids found")
	for type in imgTypes:
		histogramedType = []
		for img in type:
			histogramedImg = [0] * k
			for descriptor in img:
				index = getPrototypeNumber(descriptor, est)
				histogramedImg[index] = histogramedImg[index] + 100./len(img)
			histogramedType.append(histogramedImg)
		histogramedImgTypes.append(histogramedType)
	#in histogramedImgTypes, we have an array of types each with an array of images represented by a bag of words
	print("Images have been converted to histograms.")
	trainingSetTypes = []
	X = []
	AllTestData = []
	typeNumber = 0
	trueTestTypes = []
	for type in histogramedImgTypes:
		trainingSetForType = []
		testSetForType = []
		numberInTraining = len(type) * percentageForTraining
		randomSet = np.random.permutation(len(type))[0:numberInTraining]
		for idx in range(len(type)):
			if idx in randomSet:
				trainingSetForType.append(type[idx])
				X.append(type[idx])
			else:
				AllTestData.append(type[idx])
				trueTestTypes.append(typeNumber)
		trainingSetTypes.append(trainingSetForType)
		typeNumber = typeNumber + 1
	#Now we have randomly selected images from all types to serve as training and test data
	print("Training sets have been created")
	numberOfTrainingImages = len(X)
	i = 0
	deciders = []
	labels = []
	trainingSet = []
	for type in trainingSetTypes:
		labels= labels + ([i] * len(type))
		i = i + 1
		trainingSet = trainingSet + type
	decider = train(trainingSet, labels)
	predictions = predict(AllTestData, decider)
	for i in range(len(trainingSetTypes)):
		precision, recall, thresholds = precision_recall_curve(trueTestTypes, predictions, pos_label=i)
		plt.plot(recall, precision, label='Precision-recall curve of class {0})'.format(i))	
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Extension of Precision-Recall curve to multi-class')
	plt.legend(loc="lower right")
	plt.show()
	print("Average precision is:")
	if len(trainingSetTypes) > 2:
		print(precision_score(trueTestTypes, predictions, average='macro', sample_weight=None))
	else:
		print(precision_score(trueTestTypes, predictions, average='binary', sample_weight=None))
	return predictions
	'''for decider in deciders:
		print("Getting scores for all data")
		predictions.append(predict(AllTestData, decider))
	actualPredictions = []
	actualLabelsSize = len(predictions[0])
	for predIdx in range(actualLabelsSize):
		guess = 0
		max = -1.5
		currentClass = 0
		for prediction in predictions:
			if prediction[predIdx] > max:
				max = prediction[predIdx]
				guess = currentClass
			currentClass = currentClass + 1
		actualPredictions.append(guess)
	return deciders, actualPredictionsa'''
		
	
				
			
	