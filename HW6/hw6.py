import cv2
import numpy as np
import maxflow
from itertools import product
from scipy import signal
from scipy import ndimage

class segmenter:

	def __init__(self):
		self.drawing = False
		self.backgroundMode = True 
		self.background = []
		self.foreground = []
		self.backgroundPts = set()
		self.foregroundPts = set()
		self.img = cv2.imread('butterfly.jpg')
		self.backgroundDensity = np.zeros((256,256,256))
		self.foregroundDensity = np.zeros((256,256,256))
		self.hasCalculated = False
	
	def addToSet(self,x,y):
		if self.backgroundMode:
			self.background.append(self.img[y][x])
			self.backgroundPts.add((x, y))
		else:
			self.foreground.append(self.img[y][x])
			self.foregroundPts.add((x, y))

	def mouseEvent(self,event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.addToSet(x,y)
		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
		elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
			self.addToSet(x,y)

	def getSegment(self, graph, nodes):
		segmentation = np.zeros(self.img.shape)
		for i in range(self.img.shape[0]):
			for ii in range(self.img.shape[1]):
				segmentation[i][ii] = graph.get_segment(nodes[i][ii])
		return segmentation
	
	def dist(self, px1, px2):
		return np.linalg.norm(px1 - px2)
			
	
	def getNLinkWeight(self, i, ii, first, second):
		return self.getNpExp(self.img[i][ii], self.img[i+first][ii+second])
	
	def getNpExp(self, px1, px2):
		return np.exp(-(self.dist(px1, px2)/50))
	
	def getTWeights(self, px1):
		i = px1[0]
		ii = px1[1]
		iii = px1[2]
		if not self.hasCalculated:
			for px2 in self.background:
				a = px2[0]
				b = px2[1]
				c = px2[2]
				self.backgroundDensity[a][b][c] = self.backgroundDensity[a][b][c] + 1./len(self.background)
			for px2 in self.foreground:
				a = px2[0]
				b = px2[1]
				c = px2[2]
				self.foregroundDensity[a][b][c] = self.foregroundDensity[a][b][c] + 1./len(self.foreground)
			self.foregroundDensity = ndimage.filters.gaussian_filter(self.foregroundDensity, 25)
			self.backgroundDensity = ndimage.filters.gaussian_filter(self.backgroundDensity, 25)
			self.hasCalculated = True
		total = self.backgroundDensity[i][ii][iii] + self.foregroundDensity[i][ii][iii]
		if total == 0:
			return .5, .5
		returnvalue = self.backgroundDensity[i][ii][iii]/(total)
		return (returnvalue), ((1-returnvalue))
		
	def buildGraph(self):
		g = maxflow.Graph[float]()
		nodes = g.add_grid_nodes((self.img.shape[0], self.img.shape[1]))
		for i in range(self.img.shape[0] - 1):
			for ii in range(self.img.shape[1] - 1):
				val = self.getNLinkWeight(i,ii, 0, 1)
				g.add_edge(nodes[i][ii], nodes[i+1][ii], val, val)
				val = self.getNLinkWeight(i,ii, 1, 0)
				g.add_edge(nodes[i][ii], nodes[i][ii+1], val, val)
		for i in range(self.img.shape[0]):
			for ii in range(self.img.shape[1]):
				if (ii,i) in self.backgroundPts:
					background, foreground = 10000000, 0
				elif (ii, i) in self.foregroundPts:
					background, foreground = 0, 10000000
				else:
					background, foreground = self.getTWeights(self.img[i][ii])
				g.add_tedge(nodes[i][ii], background, foreground)
		flow = g.maxflow() #is this necessary
		return self.getSegment(g, nodes)
		
	def test(self):
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', self.mouseEvent)
		while(1):
			cv2.imshow('image', self.img)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('m'):
				self.backgroundMode = not self.backgroundMode
			elif k == ord('d'):
				break
		cv2.destroyAllWindows()
		result = self.buildGraph()
		return result