import skimage.color as C
import numpy as np
import scipy.signal as SIG
import matplotlib.pyplot as plt

def UniformBlurring(IMG):
	Kernel = np.asarray([[1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9]])
	NewImage = SIG.convolve(IMG, Kernel, mode='same')
	return NewImage

def gradientY(inputImg):
	img = C.rgb2gray(inputImg)
	img = UniformBlurring(img)
	k = np.asarray([[1,2,1], [0, 0, 0], [-1, -2, -1]])
	return SIG.convolve(img, k, mode="same")
	
def gradientX(inputImg):
	img = C.rgb2gray(inputImg)
	img = UniformBlurring(img)
	kotherway = np.asarray([[1,0,-1], [2,0,-2], [1,0,-1]])
	return SIG.convolve(img, kotherway, mode="same")
	
def gradientMagnitude(inputImg, threshold):
	resultOneWay = gradientY(inputImg)
	resultOtherWay = gradientX(inputImg)
	resultingGradMag = np.sqrt(resultOneWay ** 2 + resultOtherWay ** 2)
	return resultingGradMag > threshold
	
def gradientOrientation(inputImg):
	gradY = gradientY(inputImg)
	gradX = gradientX(inputImg)
	return np.arctan2(gradY, gradX)
	
def laPlacian(inputImg):
	img = C.rgb2gray(inputImg)
	img = UniformBlurring(img)
	k = np.asarray([[0,-1,0],[-1, 4, -1],[0,-1,0]])
	return SIG.convolve(img, k, mode="same")
	
def laPlacianCrossings(inputImg):
	laPlace = laPlacian(inputImg)
	result = np.asarray(np.zeros(laPlace.shape, np.uint8))
	for i in range(1, laPlace.shape[0] - 1):
		for j in range(1, laPlace.shape[1] - 1):
			if laPlace[i][j] * laPlace[i+1][j] < 0 or laPlace[i][j] * laPlace[i][j+1] < 0:
				result[i][j] = 2
	return result>1

def featurePoints(img):
	return laPlacianCrossings(img) & gradientMagnitude(img, .2025)
	
def featurePointsForHough(img):
	return laPlacianCrossings(img) & gradientMagnitude(img, .5625)
	
def dist(x1, y1, x2, y2):
	diffX = (x1 - x2)**2
	diffY = (y1 - y2)**2
	return np.sqrt(diffX + diffY)
	
def meetsCriteria(x, y, radian, accumulator, originalMax):
	myValue = accumulator[x][y]
	for i in range(x-1, x+2):
		for j in range(y-1, y+2):
			if i == x and j == y:
				continue
			if accumulator[i][j]>myValue:
				return False
	threshold = .4
	if radian>32:
		threshold = threshold*1.9
	if myValue  >= threshold*32./radian*originalMax:
		return True
	elif (x<radian or x+radian>accumulator.shape[0]) or (y<radian or y+radian>accumulator.shape[1]):
			if myValue >= threshold*32./radian * originalMax * .75:
				return True
			else:
				return False
	else:
		return False
	
	
def generalCircles(img, rad):
	red = img[:,:,0]
	green = img[:,:,1]
	blue = img[:,:,2]
	features = featurePointsForHough(red) | featurePointsForHough(green) | featurePointsForHough(blue)
	accumulator = np.asarray(np.zeros((features.shape[0]+rad*2, features.shape[1]+rad*2)))
	gradYRed = gradientY(red)
	gradXRed = gradientX(red)
	gradYBlue = gradientY(blue)
	gradXBlue = gradientX(blue)
	gradYGreen = gradientY(green)
	gradXGreen = gradientX(green)
	magnitudeRed = np.sqrt(gradYRed**2 + gradXRed**2)
	magnitudeBlue = np.sqrt(gradYBlue**2 + gradXBlue**2)
	magnitudeGreen = np.sqrt(gradYGreen**2 + gradXGreen**2)
	for i in range(features.shape[0]):
		for j in range(features.shape[1]):
			if(features[i][j]):
				if magnitudeRed[i][j] >= magnitudeBlue[i][j] and magnitudeRed[i][j] >= magnitudeGreen[i][j]:
					normalizedGradientX = gradXRed[i][j]/magnitudeRed[i][j]
					normalizedGradientY = gradYRed[i][j]/magnitudeRed[i][j]
				elif magnitudeBlue[i][j] >= magnitudeRed[i][j] and magnitudeBlue[i][j] >= magnitudeGreen[i][j]:
					normalizedGradientX = gradXBlue[i][j]/magnitudeBlue[i][j]
					normalizedGradientY = gradYBlue[i][j]/magnitudeBlue[i][j]
				else:
					normalizedGradientX = gradXGreen[i][j]/magnitudeGreen[i][j]
					normalizedGradientY = gradYGreen[i][j]/magnitudeGreen[i][j]
				theta = 0
				pointSet = set()
				while theta< np.pi*2:
					k=rad*np.cos(theta)+i + rad
					l=rad*np.sin(theta)+j + rad
					theta = theta + np.pi/16
					if (k,l) in pointSet or k>=accumulator.shape[0] or l>=accumulator.shape[1] or k<0 or l<0:
						continue
					pointSet.add((k,l))
					ptOne = k - rad
					ptTwo = l - rad
					distance = dist(ptOne, ptTwo, i, j)
					distDiff = np.abs(distance - rad)
					yComp = i - ptOne
					xComp = j - ptTwo
					vote = (xComp * normalizedGradientX + yComp * normalizedGradientY)
					accumulator[k][l] = accumulator[k][l] + abs(vote)
	accumulator = UniformBlurring(accumulator)
	copy = accumulator
	originalMax = accumulator.max()
	print(str(originalMax))
	print("With radius " + str(rad) + ":")
	for x in range(1, accumulator.shape[0] - 1):
		for y in range(1, accumulator.shape[1] - 1):
			if meetsCriteria(x, y, rad, accumulator, originalMax):
				print("("+str(y-rad)+", "+str(x-rad)+")")
				xMax = min(x+rad/4, accumulator.shape[0])
				yMax = min(y+rad/4, accumulator.shape[1])
				xMin = max(0, x-rad/4)
				yMin = max(0, y-rad/4)
				for i in range(xMin, xMax):
					for j in range(yMin, yMax):
						accumulator[i][j] = 0
				plt.axes()
				circle = plt.Circle((y-rad, x-rad), radius=rad, ec='y')
				plt.gca().add_patch(circle)
	plt.axis('scaled')
	ax=plt.gca()
	ax.set_ylim(ax.get_ylim()[::-1])
	plt.show()
	return copy		
							

def allCircles(img):
	generalCircles(img, 48)
	generalCircles(img, 32)
	generalCircles(img, 16)
			
	