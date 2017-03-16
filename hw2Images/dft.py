import numpy
import matplotlib.pyplot as plt
import operator
def dft(data):
	M = len(data)
	F = [0 + 0j for i in range(M)]
	twoPiOverM = 2*numpy.pi/M
	for u in range(M):
		twoPiUOverM = twoPiOverM * u
		for x in range(M):
			twoPiUXOverM = twoPiUOverM * x
			F[u] = F[u] + data[x] * numpy.cos(twoPiUXOverM)
			F[u] = F[u] - (data[x] * numpy.sin(twoPiUXOverM))*1j
		F[u] = F[u]/M
	for u in range(M):
		if numpy.abs(F[u].imag) < .000000000001: #to get rid of near-zero values
			F[u] = F[u] - (F[u].imag*1j)
		if numpy.abs(F[u].real) < .000000000001: #to get rid of near-zero values
			F[u] = F[u] - F[u].real
	return F
	
def sinWave(s, N):
	f = [numpy.sin(2*numpy.pi*s*t/N) for t in range(N)]
	return f
	
def cosWave(s, N):
	f = [numpy.cos(2*numpy.pi*s*t/N) for t in range(N)]
	return f
	
def additiveWave(s, c):
	return s*numpy.asarray(sinWave(8, 128)) + c*numpy.asarray(cosWave(8,128))

def plotReal(arr):
	plt.plot(real(arr))
	
def real(arr):
	return [arr[i].real for i in range(len(arr))]
	
def plotImag(arr):
	plt.plot(imag(arr))

def imag(arr):
	return [arr[i].imag for i in range(len(arr))]
	
def plotMagnitude(arr):
	plt.plot(magnitude(arr))
	
def magnitude(arr):
	return [numpy.sqrt(arr[i].imag * arr[i].imag + arr[i].real * arr[i].real) for i in range(len(arr))]

def plotPowerSpectrum(arr):
	plt.plot(powerSpectrum(arr))

def powerSpectrum(arr):
	return [arr[i].imag * arr[i].imag + arr[i].real * arr[i].real for i in range(len(arr))]
	
def divide(x, y):
	if y == 0 or x==0:
		return 0
	else: 
		return x/y

def plotPhase(arr):
	plt.plot(phase(arr))
	
def phase(arr):
	return [numpy.arctan(divide(arr[i].imag, arr[i].real)) for i in range(len(arr))]
	
def findFreq(arr, n):
	arr = dft(arr)
	magnitudes = magnitude(arr)
	a = len(arr) - 1 
	b = 1
	dict = {}
	while b<=a:
		dict[b] = magnitudes[b]
		b = b + 1
		a = a - 1
	sorted_magnitudes = sorted(dict.items(), key=lambda x: x[1])
	x = len(sorted_magnitudes) - 1
	for i in range(n):
		print(sorted_magnitudes[x][0])
		x = x - 1
		
def findTransferFunction(input, output):
	inputFT = numpy.asarray(numpy.fft.fft(input))
	outputFT = numpy.asarray(numpy.fft.fft(output))
	for idx in range(len(inputFT)):
		i = inputFT[idx]
		if i.real == 0 and i.imag == 0:
			outputFT[idx] = 0 + 0j
			inputFT[idx] = 1 + 0j
	hU = outputFT/inputFT
	return hU
	
	
	