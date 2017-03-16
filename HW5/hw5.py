import cv2
import numpy as np
import random
from itertools import product

def getKeyPointsAndDescriptors(img):
	sift = cv2.SIFT()
	return sift.detectAndCompute(img, None)
	
def drawKeyPoints(imgName, kp1):
	img = cv2.imread(imgName)
	return cv2.drawKeypoints(img, kp1)
	
def getMatches(img1Name, img2Name):
	kp1, des1 = getKeyPointsAndDescriptors(img1Name)
	kp2, des2 = getKeyPointsAndDescriptors(img2Name)
	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	matches2 = bf.knnMatch(des2, des1, k=2)
	# Apply ratio test
	good1 = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good1.append((kp1[m.queryIdx], kp2[m.trainIdx]))
	good2 = []
	for m, n in matches2:
		if m.distance < .75*n.distance:
			good2.append((kp1[m.trainIdx], kp2[m.queryIdx]))
	finalMatches = []
	for a in good1:
		if a in good2:
			finalMatches.append(a)
	return finalMatches

def homographyFits(homography, match):
	homogeneous1 = [[match[0].pt[0]], [match[0].pt[1]], [1]]
	pointPrime = calculatePointPrime(homography, homogeneous1)
	homogeneous2 = [[match[1].pt[0]], [match[1].pt[1]], [1]]
	dist = np.linalg.norm(pointPrime - homogeneous2)
	return dist<1
	
def RANSAC(matches):
	maxSizeOfSet = 0
	finalH = None
	iterations = 15
	for k in range(iterations):
		i = set()
		while len(i)<4:
			i.add(random.randint(0,len(matches)-1))
		x = []
		xPrime = []
		y = []
		yPrime = []
		for idx in i:
			x.append(matches[idx][0].pt[0])
			y.append(matches[idx][0].pt[1])
			xPrime.append(matches[idx][1].pt[0])
			yPrime.append(matches[idx][1].pt[1])
		try:
			h = computeHomography(x, y, xPrime, yPrime)
		except np.linalg.linalg.LinAlgError as err:
			if 'Singular matrix' in err.message:
				k = k-1
				continue
			else:
				raise
		sizeOfSet = 0
		for match in matches:
			if homographyFits(h, match):
				sizeOfSet = sizeOfSet + 1
		if sizeOfSet > maxSizeOfSet:
			finalH = h
			maxSizeOfSet = sizeOfSet
	#print maxSizeOfSet
	#print len(matches)
	return finalH
	
		
def computeHomography(x, y, xPrime, yPrime):
	b = [xPrime[0], yPrime[0], xPrime[1], yPrime[1], xPrime[2], yPrime[2], xPrime[3], yPrime[3]]
	a = [
		[x[0], y[0], 1, 0, 0, 0, -xPrime[0]*x[0], -xPrime[0]*y[0]],
		[0,0,0,x[0],y[0],1,-yPrime[0]*x[0], -yPrime[0]*y[0]],
		[x[1], y[1], 1, 0, 0, 0, -xPrime[1]*x[1], -xPrime[1]*y[1]],
		[0,0,0,x[1],y[1],1,-yPrime[1]*x[1], -yPrime[1]*y[1]],
		[x[2], y[2], 1, 0, 0, 0, -xPrime[2]*x[2], -xPrime[2]*y[2]],
		[0,0,0,x[2],y[2],1,-yPrime[2]*x[2], -yPrime[2]*y[2]],
		[x[3], y[3], 1, 0, 0, 0, -xPrime[3]*x[3], -xPrime[3]*y[3]],
		[0,0,0,x[3],y[3],1,-yPrime[3]*x[3], -yPrime[3]*y[3]]
		]
	Hs = np.linalg.solve(a,b)
	homography = np.asarray( [
		[Hs[0], Hs[1], Hs[2]],
		[Hs[3], Hs[4], Hs[5]],
		[Hs[6], Hs[7], 1]
	])
	return homography
	
def calculatePointPrime(homography, homogeneousCoord):
	result = np.dot(homography, homogeneousCoord)
	return result/result[2]
	
def warpImage(img, homography, baseimg, h2, img2):
	rows,cols,ch = img.shape
	homogeneousBottomCorner = [[cols], [rows], [1]]
	tbc = calculatePointPrime(homography, homogeneousBottomCorner)
	#print(tbc)
	homogeneousBottomCorner = [[0], [rows], [1]]
	tbc2 = calculatePointPrime(homography, homogeneousBottomCorner)
	#print(tbc2)
	homogeneousTopCorner = [[cols], [0], [1]]
	ttc = calculatePointPrime(homography, homogeneousTopCorner)
	#print(ttc)
	homogeneousTopCorner = [[0], [0], [1]]
	ttc2 = calculatePointPrime(homography, homogeneousTopCorner)
	#print(ttc2)
	minFirst = int(min(tbc[0][0], tbc2[0][0], ttc[0][0], ttc2[0][0], 0))
	maxFirst = int(max(tbc[0][0], tbc2[0][0], ttc[0][0], ttc2[0][0], baseimg.shape[1], img.shape[1]))
	minSecond = int(min(tbc[1][0], tbc2[1][0], ttc[1][0], ttc2[1][0], 0))
	maxSecond = int(max(tbc[1][0], tbc2[1][0], ttc[1][0], ttc2[1][0], baseimg.shape[0], img.shape[0]))
	
	rows,cols,ch = img2.shape
	homogeneousBottomCorner = [[cols], [rows], [1]]
	tbcNow = calculatePointPrime(h2, homogeneousBottomCorner)
	#print(tbc)
	homogeneousBottomCorner = [[0], [rows], [1]]
	tbc2 = calculatePointPrime(h2, homogeneousBottomCorner)
	#print(tbc2)
	homogeneousTopCorner = [[cols], [0], [1]]
	ttc = calculatePointPrime(h2, homogeneousTopCorner)
	#print(ttc)
	homogeneousTopCorner = [[0], [0], [1]]
	ttc2 = calculatePointPrime(h2, homogeneousTopCorner)
	minFirst = int(min(tbcNow[0][0], tbc2[0][0], ttc[0][0], ttc2[0][0], 0, minFirst))
	maxFirst = int(max(tbcNow[0][0], tbc2[0][0], ttc[0][0], ttc2[0][0], img2.shape[1], maxFirst))
	minSecond = int(min(tbcNow[1][0], tbc2[1][0], ttc[1][0], ttc2[1][0], 0, minSecond))
	maxSecond = int(max(tbcNow[1][0], tbc2[1][0], ttc[1][0], ttc2[1][0], img2.shape[0], maxSecond))
	dimOne = int(maxFirst-minFirst)+3
	dimTwo = int(maxSecond-minSecond)+3
		
	dst = np.zeros((dimTwo,dimOne, 3), np.uint8)
	M = np.float32([[1,0,-minFirst],[0,1,-minSecond], [0,0,1]])
	dst = cv2.warpPerspective(img, homography, (dst.shape[1], dst.shape[0]))
	dst = cv2.warpPerspective(dst, M, (dst.shape[1], dst.shape[0]))
	dst2 = cv2.warpPerspective(img2, h2, (dst.shape[1], dst.shape[0]))
	dst2 = cv2.warpPerspective(dst2, M, (dst.shape[1], dst.shape[0]))
	tbcNow = tbcNow/2
	h, w, d = dst.shape
	for (i,j) in product(range(h), range(w)):
		value = dst[i][j]
		value2 = dst2[i][j]
		oneHasSome = (value.any() != 0)
		twoHasSome = (value2.any() != 0)
		if oneHasSome and twoHasSome:
			differenceInX = np.abs(i - tbcNow[1][0])
			differenceInY = np.abs(j - tbcNow[0][0])
			totalDifference = differenceInX + differenceInY
			penaltyImg = (totalDifference / (img2.shape[0] + img2.shape[1]))
			if penaltyImg > 1:
				penaltyImg = 1
			elif penaltyImg < 0:
				penaltyImg = 0
				penaltyOtherImg = 1 - penaltyImg
				dst[i][j] = penaltyOtherImg*value + penaltyImg*value2
		elif twoHasSome:
			dst[i][j] = value2
	firstBaseImageCenter = baseimg.shape[0]/2.
	secondBaseImageCenter = baseimg.shape[1]/2.
	h, w, d = baseimg.shape
	for (i,j) in product(range(h), range(w)):
		firstidx = i-minSecond
		secondidx = j-minFirst
		value = dst[firstidx][secondidx]
		if (value.any() != 0):
			differenceInX = np.abs(i - firstBaseImageCenter)
			differenceInY = np.abs(j - secondBaseImageCenter)
			totalDifference = differenceInX + differenceInY
			penaltyImg = (totalDifference / (baseimg.shape[0] + baseimg.shape[1]))
			penaltyBaseImg = 1 - penaltyImg
			dst[firstidx][secondidx] = penaltyBaseImg * baseimg[i][j] + penaltyImg * value
		else:
			dst[firstidx][secondidx] = baseimg[i][j]
	invHomography = np.linalg.inv(homography)
	invHomography2 = np.linalg.inv(h2)
	h, w, d = dst.shape
	for (i,j) in product(range(h), range(w)):
		if (dst[i][j].any() != 0):
			continue
		homogeneousPoint = [[j + minFirst], [i + minSecond], [1]]
		resultingPoint = calculatePointPrime(invHomography, homogeneousPoint)
		firstidx = resultingPoint[1][0]
		secondidx = resultingPoint[0][0]
		if firstidx<img.shape[0] and secondidx<img.shape[1] and firstidx>=0 and secondidx>=0:
			dst[i][j] = img[firstidx][secondidx]
		resultingPoint = calculatePointPrime(invHomography2, homogeneousPoint)
		firstidx = resultingPoint[1][0]
		secondidx = resultingPoint[0][0]
		if firstidx<img2.shape[0] and secondidx<img2.shape[1] and firstidx>=0 and secondidx>=0:
			dst[i][j] = img2[firstidx][secondidx]
	return dst
	
def runTest():
	img1 = cv2.imread('campus2.png')
	img2 = cv2.imread('campus1.png')
	img3 = cv2.imread('campus3.png')
	m = getMatches(img1, img2)
	m2 = getMatches(img3, img2)
	m3 = getMatches(img3, img1)
	votesForOne = len(m) + len(m3)
	votesForTwo = len(m) + len(m2)
	votesForThree = len(m3) + len(m2)
	if votesForOne >= votesForTwo and votesForOne >= votesForThree:
		temp = img2
		img2 = img1
		img1 = temp
		h = np.linalg.inv(RANSAC(m))
		h2 = RANSAC(m3)
	elif votesForThree >= votesForOne and votesForThree >= votesForTwo:
		temp = img3
		img3 = img2
		img2 = temp
		h = np.linalg.inv(RANSAC(m3))
		h2 = np.linalg.inv(RANSAC(m2))
	else:
		h = RANSAC(m)
		h2 = RANSAC(m2)
	result = warpImage(img1, h, img2, h2, img3)
	cv2.imshow('image', result)
		