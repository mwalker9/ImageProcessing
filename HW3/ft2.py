import numpy as np
import skimage.color as C
import matplotlib.pyplot as pyplot
def generateSinY(s, N, x, y): #every row has different values
	shape = (x,y)
	result = np.zeros(shape, np.uint8)
	for i in range(x):
		for j in range(y):
			result[i][j] = (np.sin(2*np.pi*s*i/N) + 1) * 127.5
	return result
	
def generateSinX(s, N, x, y): #every column has different values
	shape = (x,y)
	result = np.zeros(shape, np.uint8)
	for i in range(x):
		for j in range(y):
			result[i][j] = (np.sin(2*np.pi*s*j/N) + 1) * 127.5
	return result
	
def dispMagnitude(img):
	Fu = np.fft.fft2(img)
	Fu = np.fft.fftshift(Fu)
	pyplot.imshow(abs(Fu), cmap="gray")
	
def gaussian(x, mu, sig):
	return np.exp(-np.power(x - mu, 2.)/ (2 * np.power(sig, 2.)))
	
def oneDLowPassFilter(data):
	Hu = np.zeros(len(data), np.float64)
	for i in range(len(Hu)):
		if i < len(Hu)/2:
			Hu[i] = gaussian(i, 0, len(data)/20)
		else:
			Hu[i] = gaussian(len(Hu)-i, 0, len(data)/20)
	Fu = np.fft.fft(data)
	return np.abs(np.fft.ifft(Hu * Fu))

def gaussianTwoD(x, y, mux, muy, sigx, sigy):
		return np.exp(-(((x-mux)**2./(2*sigx**2.)) + ((y-muy)**2./(2*sigy**2.))))

def twoDLowPassFilter(img):
	Hu = np.zeros(img.shape, np.float64)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			Hu[i][j] = gaussianTwoD(i, j, img.shape[0]/2, img.shape[1]/2, img.shape[0]/4, img.shape[1]/4)
	Fu = np.fft.fft2(img)
	Fu = np.fft.fftshift(Fu)
	return np.abs(np.fft.ifft2(np.fft.ifftshift(Hu * Fu)))

def twoDHighPassFilter(img):
	Hu = np.zeros(img.shape, np.float64)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			Hu[i][j] = 1 - gaussianTwoD(i, j, img.shape[0]/2, img.shape[1]/2, img.shape[0]/4, img.shape[1]/4)
	Fu = np.fft.fft2(img)
	Fu = np.fft.fftshift(Fu)
	return np.abs(np.fft.ifft2(np.fft.ifftshift(Hu * Fu)))
	
def nineByNineBlurring(inputIMG):
	if np.ndim(inputIMG) == 3:
		Ft = np.double(C.rgb2gray(inputIMG)/255.0)[:][:]
	else:
		Ft = np.double(inputIMG/255.0)
	kernel = np.zeros(Ft.shape, np.float64)
	numForPaddingX = int((Ft.shape[0] - 9) / 2)
	numForPaddingY = int((Ft.shape[1] - 9) / 2)
	for i in range(inputIMG.shape[0]):
		for j in range(inputIMG.shape[1]):
			if i >= numForPaddingX and i<numForPaddingX+9 and j >= numForPaddingY and j<numForPaddingY+9:
				kernel[i][j] = 1/81.0
			else:
				kernel[i][j] = 0
	Fu = np.fft.fft2(Ft)
	Hu = np.fft.fft2(np.fft.fftshift(kernel))
	return np.abs(np.fft.ifft2(Fu*Hu))

def divide(a, b):
	if b == 0:
		return 0
	else:
		return np.double(np.double(a)/b)

def phaseImg(img):
	result = np.zeros(img.shape, np.float64)
	for i in range(img.shape[0]):
		result[i] = phase(img[i][:])
	return result
		
def phase(arr):
	return [np.arctan2(arr[i].imag, arr[i].real) for i in range(len(arr))]

def showTwoImages(img1, img2, figInt):
	pyplot.figure(figInt)
	pyplot.imshow(img1, cmap="gray")
	pyplot.figure(figInt + 1)
	pyplot.imshow(img2, cmap="gray")

def partD(img1, img2):
	original1 = img1
	original2 = img2
	img1 = C.rgb2gray(img1)
	img2 = C.rgb2gray(img2)
	bU = np.fft.fft2(img1)
	gU = np.fft.fft2(img2)
	bMag = abs(bU)
	gMag = abs(gU)
	bPhase = phaseImg(bU)
	gPhase = phaseImg(gU)
	bMagGPhase = np.zeros(bMag.shape, np.complex)
	for i in range(bMag.shape[0]):
		for j in range(bMag.shape[1]):
			bMagGPhase[i][j] = (bMag[i][j] * np.cos(gPhase[i][j])) + (bMag[i][j] * np.sin(gPhase[i][j])) * 1j
	gMagBPhase = np.zeros(gMag.shape, np.complex)
	for i in range(gMag.shape[0]):
		for j in range(gMag.shape[1]):
			gMagBPhase[i][j] = (gMag[i][j] * np.cos(bPhase[i][j])) + (gMag[i][j] * np.sin(bPhase[i][j])) * 1j
	bMagGPhaseInv = np.fft.ifft2(bMagGPhase)
	gMagBPhaseInv = np.fft.ifft2(gMagBPhase)
	showTwoImages(abs(bMagGPhaseInv), abs(gMagBPhaseInv), 1)
	img1 = original1
	img2 = original2
	bHighPass = np.fft.fft2(twoDHighPassFilter(img1))
	gHighPass = np.fft.fft2(twoDHighPassFilter(img2))
	bLowPass = np.fft.fft2(twoDLowPassFilter(img1))
	gLowPass = np.fft.fft2(twoDLowPassFilter(img2))
	showTwoImages(abs(np.fft.ifft2(bHighPass + gLowPass)), abs(np.fft.ifft2(bLowPass + gHighPass)), 3)

def partE(img):
	img = C.rgb2gray(img)
	Fu = np.fft.fft2(img)
	idxX = 0
	idxY = 0
	diff = -1.
	shouldBe = -1
	currMag = -1
	for i in range(1, Fu.shape[0] / 2):
		for j in range(1, Fu.shape[1] / 2):
			value = abs(Fu[i][j])
			top = abs(Fu[i-1][j])
			bottom = abs(Fu[i+1][j])
			right = abs(Fu[i][j+1])
			left = abs(Fu[i][j-1])
			if value == 0:
				continue
			currDiff = np.mean([abs(value-top), abs(value-bottom), abs(value-right), abs(value-left)])
			if currDiff>diff:
				diff = currDiff
				idxX = i
				idxY = j
				shouldBe = np.mean([top, bottom, right, left])
				currMag = value
	negX = Fu.shape[0] - idxX
	negY = Fu.shape[1] - idxY
	Fu[idxX][idxY] = (shouldBe/currMag) * Fu[idxX][idxY].real + ((shouldBe/currMag) * Fu[idxX][idxY].imag * 1j)
	idxX = negX
	idxY = negY
	Fu[idxX][idxY] = (shouldBe/currMag) * Fu[idxX][idxY].real + ((shouldBe/currMag) * Fu[idxX][idxY].imag * 1j)
	return np.abs(np.fft.ifft2(Fu))
	
	
	
	

	
		
				