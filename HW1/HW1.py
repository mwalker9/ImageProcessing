import numpy
import scipy.misc as SCI
import skimage.color as C
import scipy.signal as SIG
import math
def GrayScale(IMG):
	GreyedIMG = numpy.zeros(IMG.shape, 'uint8')
	GreyedIMG[:,:,0] = .299 * IMG[:,:,0] + .587 * IMG[:,:,1] + .114 * IMG[:,:,2]
	GreyedIMG[:,:,1] = GreyedIMG[:,:,0]
	GreyedIMG[:,:,2] = GreyedIMG[:,:,0]
	return GreyedIMG

def BrightnessAdjustment(IMG, amtToAdjust):
	AdjustedIMG = numpy.zeros(IMG.shape, 'uint8')
	for i in range(IMG.shape[0]):
		for ii in range(IMG.shape[1]):
			temp = (int)(IMG[i,ii] + amtToAdjust)
			if temp>=255:
				AdjustedIMG[i,ii] = 255
			elif temp<=0:
				AdjustedIMG[i,ii] = 0
			else:
				AdjustedIMG[i,ii] = temp
	return AdjustedIMG
	
def UniformBlurring(IMG):
	Kernel = numpy.asarray([[1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9]])
	NewImage = numpy.zeros(IMG.shape, 'uint8')
	NewImage[:,:] = SIG.convolve(IMG[:,:], Kernel, mode='same')
	return NewImage
	
def Sharpening(IMG):
	Kernel = numpy.asarray([[0, -1, 0], [-1, 5, -1], [0,-1,0]])
	NewImage = numpy.zeros(IMG.shape, 'int16')
	NewImage[:,:] = SIG.convolve(IMG[:,:], Kernel, mode='same')
	FinalImage = numpy.zeros(NewImage.shape, 'uint8')
	for i in range(NewImage.shape[0]):
		for ii in range(NewImage.shape[1]):
			if NewImage[i,ii] >= 255:
				FinalImage[i,ii] = 255
			elif NewImage[i,ii] <= 0:
				FinalImage[i,ii] = 0
			else:
				FinalImage[i,ii] = NewImage[i,ii]
	return FinalImage
	
def EdgeDetection(IMG):
	xKernel = numpy.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	xIMG = numpy.zeros(IMG.shape, 'float64')
	xIMG[:,:] = SIG.convolve(IMG[:,:], xKernel, mode='same')/ 8.0
	yKernel = numpy.asarray([[-1,-2,-1], [0,0,0], [1,2,1]])
	yIMG = numpy.zeros(IMG.shape, 'float64')
	yIMG[:,:] = SIG.convolve(IMG[:,:], xKernel, mode='same')/ 8.0
	NewImage = (xIMG ** 2 + yIMG ** 2) ** .5
	FinalImage = numpy.zeros(NewImage.shape, 'uint8')
	for i in range(NewImage.shape[0]):
		for ii in range(NewImage.shape[1]):
			FinalImage[i,ii] = NewImage[i,ii]
	return FinalImage
	
def MedianFilter(IMG):
	resultingIMG = numpy.zeros(IMG.shape, 'uint8')
	for i in range(1, IMG.shape[0] - 1):
		numSet = []
		numSet.append(IMG[i+1,2])
		numSet.append(IMG[i-1,2])
		numSet.append(IMG[i,2])
		numSet.append(IMG[i,1])
		numSet.append(IMG[i-1,1])
		numSet.append(IMG[i+1,1])
		numSet.append(IMG[i+1,0])
		numSet.append(IMG[i-1,0])
		numSet.append(IMG[i,0])
		for ii in range(1, IMG.shape[1] - 1):
			numSet = sorted(numSet)
			resultingIMG[i,ii] = numSet[4]
			numSet.remove(IMG[i,ii-1])
			numSet.remove(IMG[i+1,ii-1])
			numSet.remove(IMG[i-1,ii-1])
			if ii + 2 < IMG.shape[1]:
				numSet.append(IMG[i,ii+2])
				numSet.append(IMG[i+1,ii+2])
				numSet.append(IMG[i-1,ii+2])
	resultingIMG[0, :] = IMG[0, :]
	resultingIMG[:, 0] = IMG[:, 0]
	resultingIMG[IMG.shape[0] - 1, :] = IMG[IMG.shape[0] - 1, :]
	resultingIMG[:,IMG.shape[1] - 1] = IMG[:,IMG.shape[1] - 1]
	return resultingIMG

def HistogramEqualize(IMG):
	histogram = numpy.zeros(256, 'uint64')
	for i in range(IMG.shape[0]):
		for ii in range(IMG.shape[1]):
			histogram[IMG[i,ii]] += 1
	area = numpy.sum(histogram)
	FormulaFrontRatio = 255.0 / area
	resultingValues = []
	currentArea = 0
	for value in histogram:
		resultingValues.append((int)(FormulaFrontRatio * currentArea))
		currentArea += value
	resultingIMG = numpy.zeros(IMG.shape, 'uint8')
	for i in range(IMG.shape[0]):
		for ii in range(IMG.shape[1]):
			resultingIMG[i,ii] = resultingValues[IMG[i,ii]]
	return resultingIMG
		
		
		
		
	
	
			
			
				
	
	