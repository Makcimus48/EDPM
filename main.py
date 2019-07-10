# ================================================
#	My imports 
import model
import cv2 as mod 
# Create object:
#	A = model.Model(N>=1, S>=1, Title, </leftBoarder, step>=1/>) N - cnt of dots, S - amplitude between -S...+S,
#																 Title - name, leftBoarder - position first dot on X axis,
#																 step - range between two iterable dots
# Attribute:
#	self.arr - array of dots
#	self.N - count of dots
#	self.S - scale/ amplitude dot on Y axis
#	self.title - name of arr/ funcrion
#	self.lBorder - position first dot on X axis
#	self.rBorder = position last dot on X axis
#	self.step - range between two iterable dots
#	self.x - array of dots by X axis
#	self.currentSeed - number for function fillArrSelfRandom()
#	self.min - min namber by Y axis
#	self.max - max namber by Y axis
#	self.frequensy - frequency a generation of harmonic function A
#	self.amplitude - amplitude a generation of harmonic function f
# Function:
#	harmonic_process(self, A, f, _t) on object model
#	__minMax(self) inside function for find min/+ max
#	trend(self, type) function for 5 static graphs on object modul
#	fillArrRandom(self) default random on object modul
#	fillArrSelfRandom(self) my random on object modul
#	normalization(self) on object modul after random or selfRandom
#	shift(self, Count, lBord = 0, rBord = -1) on object modal shifts part or all of the graph
#	spikes(self, p1, p2, p3, p4) on object model: p1 <=1% - count, p2 - scatter %,
#								 p3 - reference amplitude in int <= k [k = 6], p4 - amplitude adjustment  < k and <p3
import inOut
# Create object:
#	A = inOut.inOut([obj1,obj2,...]) array of object(model)
# Attribute:
#	self.x - axis x
#	self.obj - object model
#	self.arrs - array a point array
# 	self.names - array a names for equal position a point array 
# Function:
# 	A.showHisto(self, interval) on object inOut
# 	A.showInCulum(self) on object  inOut
# 	A.inOut.inOut.simpleShow(arr, min, title, step = 1) func dont matter for what (obj or somthing else)
# 	A.multiAutoShow(self) on object inOut 
import analysis
#
# ================================================
import copy
from shutil import copyfile
from math import sin, cos, sqrt, log, floor
from scipy import signal
import PIL
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imread, imshow, imsave
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import random
import pickle
from scipy.ndimage import maximum_filter


##Подаём выборки на отрисовку в столбец inOut(arrs[...])
#pl1 = model.Model(1000, 20, "Mix")
#pl1.trend("mix")
#pl2 = model.Model(1000, 20, "Lineral x")
#pl2.trend("x")
#pl3 = model.Model(1000, 20, "Lineral -x")
#pl3.trend("-x")
#pl4 = model.Model(1000, 20, "exp")
#pl4.trend("e")
#pl5 = model.Model(1000, 20, "-exp")
#pl5.trend("-e")
#pl6 = model.Model(1000, 20, "Default random")
#pl6.fillArrRandom()
#pl6.normalization()
#pl7 = model.Model(1000, 20, "My random")
#pl7.fillArrSelfRandom()
#pl7.normalization()
#pl8 = model.Model(1000, 20, "Shift part")
#pl8.fillArrRandom()
#pl8.normalization()
#pl8.shift(20, 250, 650)
#pl9 = model.Model(1000, 20, "Spikes")
#pl9.fillArrRandom()
#pl9.normalization()
#pl9.spikes(1, 4.5, 5, 2.5)
#
#window1 = inOut.inOut([pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8,pl9])
#window1.smartShow()
#
#pl10 = model.Model(10000, 20, "Default random")
#pl10.fillArrRandom()
#
#pl11 = model.Model(10000, 20, "Default random")
#pl11.fillArrSelfRandom()
#
#window2 = inOut.inOut([pl10])
#window2.showHisto(100)
#window3 = inOut.inOut([pl11])
#window3.showHisto(100)
#
#ana = analysis.Analysis(pl11)
#ana.station(10)
#ana.statistic()
#print('M = ', ana.M)
#print('D = ', ana.D)
#print('~M = ', ana._M)
#print('~D = ', ana._D)
#print('СреднОтклон = ', ana.standartDev)
#print('СреднКвадр = ', ana.rndSquare)
#print('СреднКвадратОшибк = ', ana.rootMeanSquareErr)
#print('Ассиметрия = ', ana.asymmetry)
#print('Экцесс = ', ana.excess)
#print('СтандарОтклонение = ', ana.coefAsymm)
#print('КоэффЭкцесс = ', ana.coefExce)

#pl1 = model.Model(1000, 20)
## A - amplitude f - frequensy _t - dt
#pl1.harmonic_process(aTmp = [100], fTmp = [3])
#
#
#window = inOut.inOut([pl1])
#window.smartShow()

##=======================================
#pl2 = 
#ana = analysis.Analysis(pl2)
#ana.autocorrelation()
#window = inOut.inOut([pl2])
#window.smartShow(['sS', 'hG'],['Rxx', 'Rxx'])
#
#pl1 = model.Model(1000)
#pl1.fillArrRandom()
#pl1.normalization(20)
#pl1.child.shift(40)
#analysis.Analysis(pl1.child.child)
##=======================================


#pl1 = model.Model(1000)
#pl1.harmonic_process(aTmp = [10], fTmp = [3])
#pl1.normalization(20)
#pl1.last().antiRandom(200)
#window3 = inOut.inOut([pl1,pl1.last(), pl1.get(0)])
#window3.smartShow()#['sS','hG'],['baseArr','baseArr'])


#pl1 = model.Model(1000)
#pl1.harmonic_process(aTmp = [100], fTmp = [43])
#ana = analysis.Analysis(pl1)
#ana.autocorrelation()
#ana.furie()
#ana.furie(pl1.analys.CS)
#window3 = inOut.inOut([pl1])
#window3.smartShow(['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])

#pl1 = model.Model(1000)
#pl1.harmonic_process(aTmp = [100, 25, 50, 94, 13, 74, 66], fTmp=[43, 547, 457, 65, 34, 977, 257])
#ana = analysis.Analysis(pl1)
#ana.autocorrelation()
#ana.furie()
#ana.furie(pl1.analys.CS)
#window3 = inOut.inOut([pl1])
#window3.smartShow(['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])

#pl1 = model.Model(1000)
#pl1.fillArrRandom()
#ana = analysis.Analysis(pl1)
#ana.autocorrelation()
#ana.furie()
#ana.furie(pl1.analys.CS)
#window3 = inOut.inOut([pl1])
#window3.smartShow(['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])
#
#pl1 = model.Model(1000)
#pl1.fillArrSelfRandom()
#ana = analysis.Analysis(pl1)
#ana.autocorrelation()
#ana.furie()
#ana.furie(pl1.analys.CS)
#window3 = inOut.inOut([pl1])
#window3.smartShow(['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])
#
#pl1 = model.Model(1000)
#pl1.trend('x')
#ana = analysis.Analysis(pl1)
#ana.autocorrelation()
#ana.furie()
#ana.furie(pl1.analys.CS)
#window3 = inOut.inOut([pl1])
#window3.smartShow(['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])
#
#pl1 = model.Model(1000)
#pl1.trend('-x')
#ana = analysis.Analysis(pl1)
#ana.autocorrelation()
#ana.furie()
#ana.furie(pl1.analys.CS)
#window3 = inOut.inOut([pl1])
#window3.smartShow(['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])






#pl1 = model.Model(1000)
#pl1.trend('e')
#
#window3 = inOut.inOut([pl1])
#window3.smartShow()#['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])
#
#pl1 = model.Model(1000)
#pl1.trend('-e')
#
#window3 = inOut.inOut([pl1])
#window3.smartShow()

#pl1 = model.Model(1000)
#pl1.trend('mix')
#
#window3 = inOut.inOut([pl1])
#window3.smartShow()
############################################
## Шаблон свёртки
#pl1 = model.Model(200)
#pl1.heartSin()
#
#pl2 = model.Model(1000)
#pl2.heartSpicks()
#
#pl3 = model.Model(1000)
#pl3.convolution(pl2,pl1)
#
#window3 = inOut.inOut([pl1, pl2, pl3])
#window3.smartShow()


############pl1 = model.Model(1000)
############pl1.harmonic_process(aTmp = [130, 45, 79], fTmp = [7, 25, 56]) # Делает полигармонический процесс из 3 гармоник
############ana = analysis.Analysis(pl1)
############ana.furie()				
############pl2 = model.Model(4)
############pl2.lpf(7,128,0.002)   # Любой фильтр
############ana1 = analysis.Analysis(pl2)
############ana1.furie()				
############pl3 = model.Model(1000)
############pl3.convolution(pl1,pl2)
############pl3.arr['baseArr'] = pl3.arr['baseArr'][129:pl3.N - 144:1] 
############pl3.N = len(pl3.arr['baseArr'])
############ana2 = analysis.Analysis(pl3)
############ana2.furie()
############window3 = inOut.inOut([pl1,pl2, pl3])
############window3.smartShow(['sS','sS'],['baseArr','C'])
############
############pl1 = model.Model(1000)
############pl1.harmonic_process(aTmp = [130, 45, 79], fTmp = [7, 25, 56])
############ana = analysis.Analysis(pl1)
############ana.furie()
############pl2 = model.Model(4)
############pl2.hpf(56,128,0.002)
############ana1 = analysis.Analysis(pl2.get(1))
############ana1.furie()
############pl3 = model.Model(1000)
############pl3.convolution(pl1,pl2.get(1))
############
############pl3.arr['baseArr'] = pl3.arr['baseArr'][129:pl3.N - 144:1]
############pl3.N = len(pl3.arr['baseArr'])
############ana2 = analysis.Analysis(pl3)
############ana2.furie()
############window3 = inOut.inOut([pl1,pl2.get(1), pl3])
############window3.smartShow(['sS','sS'],['baseArr','C'])
############
############pl1 = model.Model(1000)
############pl1.harmonic_process(aTmp = [130, 45, 79], fTmp = [7, 25, 56])
############ana = analysis.Analysis(pl1)
############ana.furie()
############pl2 = model.Model(4)
############pl2.bpf(24, 27,128,0.002)
############ana1 = analysis.Analysis(pl2.get(1))
############ana1.furie()
############pl3 = model.Model(1000)
############pl3.convolution(pl1,pl2.get(1))
############
############pl3.arr['baseArr'] = pl3.arr['baseArr'][129:pl3.N - 144:1]
############pl3.N = len(pl3.arr['baseArr'])
############ana2 = analysis.Analysis(pl3)
############ana2.furie()
############window3 = inOut.inOut([pl1,pl2.get(1), pl3])
############window3.smartShow(['sS','sS'],['baseArr','C'])
############
############pl1 = model.Model(1000)
############pl1.harmonic_process(aTmp = [130, 45, 79], fTmp = [7, 25, 56])
############ana = analysis.Analysis(pl1)
############ana.furie()
############pl2 = model.Model(4)
############pl2.bsf(20, 35,128,0.002)
############ana1 = analysis.Analysis(pl2.get(1))
############ana1.furie()
############pl3 = model.Model(1000)
############pl3.convolution(pl1,pl2.get(1))
############
############pl3.arr['baseArr'] = pl3.arr['baseArr'][129:pl3.N - 144:1]
############pl3.N = len(pl3.arr['baseArr'])
############ana2 = analysis.Analysis(pl3)
############ana2.furie()
############window3 = inOut.inOut([pl1,pl2.get(1), pl3])
############window3.smartShow(['sS','sS'],['baseArr','C'])
##########################################################
# Выделение тренда путём скользящего среднего
#pl1 = model.Model(1000)
#pl1.fillArrRandom()
#pl1.normalization(80)
#pl1.child.trend('mix')
#pl1.get(1).getTrend(15)
#
#window3 = inOut.inOut([pl1,pl1.get(1), pl1.get(2)])
#window3.smartShow()
#############################################
#pl1 = model.Model(1000)
#pl1.harmonic_process(aTmp = [100], fTmp = [3])
#pl1.inerpolate()
#ana = analysis.Analysis(pl1)
##ana.autocorrelation()
##ana.furie()
##ana.furie(pl1.analys.CS)
##ana1 = analysis.Analysis(pl1.child)
##ana1.autocorrelation()
##ana1.furie()
##ana1.furie(pl1.analys.CS)
#win = inOut.inOut([pl1, pl1.child])
#win.smartShow()#['sS','sS','sS','sS'],['baseArr','C', 'reC', 'Rxx'])
##############################################

###pl1 = model.Model(150)
###pl1.harmonic_process(_t = 3.6764e-8,aTmp = [100], fTmp = [1.5e+6])
###for i in range(36):
###	pl1.arr['baseArr'][i] = pl1.arr['baseArr'][i] * sqrt(i)/6
###	pl1.arr['baseArr'][len(pl1.arr['baseArr']) - i-1] = pl1.arr['baseArr'][len(pl1.arr['baseArr']) - i-1] * sqrt(i)/6
###
###pl2 = model.Model(1000)
###pl2.reflectSpicks()
###
###pl3 = model.Model(100)
###pl3.convolution(pl2,pl1)
###pl4 = model.Model(100)
###pl4.convolution(pl3,pl1)
####ana1 = analysis.Analysis(pl1)
####ana1.furie()
####ana2 = analysis.Analysis(pl2)
####ana2.furie()
####ana3 = analysis.Analysis(pl3)
####ana3.furie()
###
####pl5 = copy.deepcopy(pl4)
####pl5.deconvolve(pl4, pl1)
####pl6 = copy.deepcopy(pl5)
####pl6.deconvolve(pl5, pl1)
####print(len(pl4.arr['C']))
####print(len(pl1.arr['C']))
####for i in range(len(pl4.arr['baseArr'])):
####	pl4.arr['C'][i] = pl4.arr['C'][i] / pl1.arr['C'][i]
###win = inOut.inOut([pl1, pl2, pl3, pl4, pl4, pl4])
###win.smartShow()#['sS','sS'],['baseArr','C'])


#pl1 = model.Model(4)
#pl1.lpf(10,128,0.002)
#ana = analysis.Analysis(pl1)
#ana.furie()
#win = inOut.inOut([pl1])
#pl1.arr['C'] = [i * pl1.N for i in pl1.arr['C']]
#win.smartShow(['sS','sS'],['baseArr','C'])
#
#pl1 = model.Model(4)
#pl1.hpf(10,128,0.002)
#ana = analysis.Analysis(pl1.get(1))
#ana.furie()
#win = inOut.inOut([pl1.get(1)])
#pl1.get(1).arr['C'] = [i * pl1.get(1).N for i in pl1.get(1).arr['C']]
#win.smartShow(['sS','sS'],['baseArr','C'])
#
#pl1 = model.Model(4)
#pl1.bpf(10,24,128,0.002)
#ana = analysis.Analysis(pl1.get(1))
#ana.furie()
#win = inOut.inOut([pl1.get(1)])
#pl1.get(1).arr['C'] = [i * pl1.get(1).N for i in pl1.get(1).arr['C']]
#win.smartShow(['sS','sS'],['baseArr','C'])
#
#pl1 = model.Model(4)
#pl1.bsf(10,24,128,0.002)
#ana = analysis.Analysis(pl1.get(1))
#ana.furie()
#win = inOut.inOut([pl1.get(1)])
#pl1.get(1).arr['C'] = [i * pl1.get(1).N for i in pl1.get(1).arr['C']]
#win.smartShow(['sS','sS'],['baseArr','C'])
###############################################
#pl1 = model.Model(1000)
#pl1.fillArrRandom()
#pl1.normalization(20)
#pl1.child.shift(40)
#pl1.child.child.antiShift()
#
#window3 = inOut.inOut([pl1,pl1.child,pl1.child.child,pl1.child.child.child])
#window3.smartShow()



def openIm (fileName):
	if fileName.split('.')[-1].lower() == 'xcr':
		# fix this later
		print('This part is not yet describe')
	elif fileName.split('.')[-1].lower() == 'dat':
		# fix this later
		print('This part is not yet describe')
	else:
		im = imread(fileName, mode="RGB")
	return im, fileName

def showInfo(imFile, imgName):
	if len(imFile.shape) == 2:
		height, width = imFile.shape
		deep = 1
	else:
		height, width, deep = imFile.shape
	minP = float('inf')
	maxP = float('-inf')
	for i in range(height):
		for j in range(width):
			tmp = imFile[i][j]
			if deep > 1:
				tmp = int(sum(tmp)/deep)
			if  tmp > maxP:
				maxP = int(tmp)
			elif tmp < minP:
				minP = int(tmp)
	if deep == 1:
		cmap = 'Gray'
	else:
		cmap = 'RGB'
	avgColum = [0] * width
	avgRow = [0] * height
	for i in range(height):
		for j in range(width):
			tmp = imFile[i][j]
			if deep > 1:
				tmp = int(sum(tmp)/deep)
			avgColum[j] += tmp
			avgRow[i] += tmp
	for i in range(width):
		avgColum[i] = int(avgColum[i]/height)
	for i in range(height):
		avgRow[i] = int(avgRow[i]/width)
	print('Изображение: ', imgName, '\nШирина: ', width, '\nВысота: ', height)
	print('[in ', cmap, ' scale] Max color: ', maxP,' Min color', minP)
	x = [i for i in range(width)]
	y = [i for i in range(height)]
	plt.subplot(1,2,1)
	plt.plot(x,avgColum)
	plt.title("Avg on colum")
	plt.subplot(1,2,2)
	plt.plot(y,avgRow)
	plt.title("Avg on row")
	plt.show()

def saveAs(imgFile, imgName):
	if type(imgFile) is str:
		imsave("stuff/"+imgFile+".jpg", imgName)
		#print("/Used/"+imgFile.split('_')[-1])
		copyfile("Used/"+imgFile.split('_')[-1], imgFile)
	else:
		imsave('fixed_'+imgName.split('/')[-1], imgFile)

def toGray(imFile):
	return np.dot(imFile[...,:3], [0.2989, 0.5870, 0.1140])

def normize(imFile, isShow = True):
	if len(imFile.shape) == 2:
		height, width = imFile.shape
		deep = 1
	else:
		height, width, deep = imFile.shape
	minP = float('inf')
	maxP = float('-inf')
	arr = copy.deepcopy(imFile)
	for i in range(height):
		for j in range(width):
			tmp = imFile[i][j]
			if deep > 1:
				tmp = int(sum(tmp)/deep)
			if  tmp > maxP:
				maxP = int(tmp)
			elif tmp < minP:
				minP = int(tmp)
			arr[i, j] = tmp
	if deep == 1:
		for i in range(imFile.shape[0]):
			for j in range(imFile.shape[1]):
				arr[i, j] = int(255 * (arr[i, j] - minP)/(maxP-minP))
	else:
		for i in range(imFile.shape[0]):
			for j in range(imFile.shape[1]):
				tmp = int(255 * (arr[i, j][0] - minP)/(maxP-minP))
				arr[i, j][0] = tmp
				arr[i, j][1] = tmp
				arr[i, j][2] = tmp
	if isShow:
		plt.subplot(1,2,1)
		plt.imshow(arr, cmap="gray")
		plt.title("Normilyzed")
		plt.subplot(1,2,2)
		plt.imshow(imFile, cmap="gray")
		plt.title("Original")
		plt.show()
	return arr

def resize(imgFile, imgName, k, mode = True): # mode true = increase false = decrease
	if len(imgFile.shape) == 2:
		h, w = imgFile.shape
		deep = 1
	else:
		h, w, deep = imgFile.shape
	if mode:
		k = k
	else:
		k = 1/k
	width = round(w * k)
	height = round(h * k)
	if deep == 1:
		arr = np.array([0 for i in range(width) for j in range(height)])
		arr = arr.reshape(height,width)
	else:
		arr = np.array([[0]*deep for i in range(width) for j in range(height)])
		arr = arr.reshape(height,width,deep)
	for j in range(width):
		for i in range(height):
			old_i=int(i/k)
			old_j=int(j/k)
			#if(old_i>=width):
			#	old_i=width-1
			#if(old_j>=height):
			#	old_j=height-1
			arr[i,j] = imgFile[old_i, old_j]
	saveAs(arr, imgName)
	plt.subplot(1,2,1)
	plt.imshow(arr, cmap="gray")
	plt.title("Near neighbor")
	plt.subplot(1,2,2)
	plt.imshow(imgFile, cmap="gray")
	plt.title("Original")
	plt.show()
	return arr


def bilinearResize(imgFile, imgName, scale, mode = True): # mode true = increase false = decrease
	if len(imgFile.shape) == 2:
		h, w = imgFile.shape
		deep = 1
	else:
		h, w, deep = imgFile.shape
	if mode:
		scale = scale
	else:
		scale = 1/scale
	width = round(w * scale)
	height = round(h * scale)
	if deep == 1:
		arr = np.array([0 for i in range(width) for j in range(height)])
		arr = arr.reshape(height,width)
	else:
		arr = np.array([[0]*deep for i in range(width) for j in range(height)])
		arr = arr.reshape(height,width,deep)
	rowScale = float(h) / float(height)
	colScale = float(w) / float(width)
	for r in range(height):
		for c in range(width):
			posY = r * rowScale 
			posX = c * colScale
			modXi = int(posX)
			modYi = int(posY)
			modXf = posX - modXi
			modYf = posY - modYi
			modXiPlusOneLim = min(modXi+1,w-1)
			modYiPlusOneLim = min(modYi+1,h-1)
			if deep == 1:
				out = 0
				bl = imgFile[modYi, modXi]
				br = imgFile[modYi, modXiPlusOneLim]
				tl = imgFile[modYiPlusOneLim, modXi]
				tr = imgFile[modYiPlusOneLim, modXiPlusOneLim]
				b = modXf * br + (1. - modXf) * bl
				t = modXf * tr + (1. - modXf) * tl
				pxf = modYf * t + (1. - modYf) * b
				out = int(pxf+0.5)
			else:
				out = []
				for chan in range(deep):
					bl = imgFile[modYi, modXi, chan]
					br = imgFile[modYi, modXiPlusOneLim, chan]
					tl = imgFile[modYiPlusOneLim, modXi, chan]
					tr = imgFile[modYiPlusOneLim, modXiPlusOneLim, chan]
					b = modXf * br + (1. - modXf) * bl
					t = modXf * tr + (1. - modXf) * tl
					pxf = modYf * t + (1. - modYf) * b
					out.append(int(pxf+0.5))
			arr[r, c] = out
	saveAs(arr, imgName)
	plt.subplot(1,2,1)
	plt.imshow(arr, cmap="gray")
	plt.title("Bilinear")
	plt.subplot(1,2,2)
	plt.imshow(imgFile, cmap="gray")
	plt.title("Original")
	plt.show()
	return arr

def negativ(imgFile, imgName):
	arr = copy.deepcopy(imgFile)
	if len(imgFile.shape) == 2:
		for i in range(imgFile.shape[0]):
			for j in range(imgFile.shape[1]):
				arr[i,j] = 255  - imgFile[i,j]
	else:
		for i in range(imgFile.shape[0]):
			for j in range(imgFile.shape[1]):
				arr[i,j][0] = 255  - imgFile[i,j][0]
				arr[i,j][1] = 255  - imgFile[i,j][1]
				arr[i,j][2] = 255  - imgFile[i,j][2]
	saveAs(arr, "ree/neg_"+imgName+".jpg")
	plt.subplot(1,2,1)
	plt.imshow(arr, cmap="gray")
	plt.title("Negative")
	plt.subplot(1,2,2)
	plt.imshow(imgFile, cmap="gray")
	plt.title("Original")
	plt.show()
	return arr

def hammaCorrect(imgFile, k):
	arr = copy.deepcopy(imgFile)
	if len(imgFile.shape) == 2:
		for i in range(imgFile.shape[0]):
			for j in range(imgFile.shape[1]):
				arr[i,j] = imgFile[i,j] ** k
	else:
		for i in range(imgFile.shape[0]):
			for j in range(imgFile.shape[1]):
				tmp = imgFile[i,j][0] ** k
				arr[i,j][0] = tmp
				arr[i,j][1] = tmp
				arr[i,j][2] = tmp
	arr = normize(arr, False)
	plt.subplot(1,2,1)
	plt.imshow(arr, cmap="gray")
	plt.title("Hamma")
	plt.subplot(1,2,2)
	plt.imshow(imgFile, cmap="gray")
	plt.title("Original")
	plt.show()
	return arr

def logorFunc(imgFile, k):
	arr = copy.deepcopy(imgFile)
	if len(imgFile.shape) == 2:
		for i in range(imgFile.shape[0]):
			for j in range(imgFile.shape[1]):
				arr[i,j] = log(1 + imgFile[i,j], k)
	else:
		for i in range(imgFile.shape[0]):
			for j in range(imgFile.shape[1]):
				tmp = log(1 + imgFile[i,j][0], k)
				arr[i,j][0] = tmp
				arr[i,j][1] = tmp
				arr[i,j][2] = tmp
	arr = normize(arr, False)
	plt.subplot(1,2,1)
	plt.imshow(arr, cmap="gray")
	plt.title("Logarifmed")
	plt.subplot(1,2,2)
	plt.imshow(imgFile, cmap="gray")
	plt.title("Original")
	plt.show()
	return arr

def addition(imgFile,y,imgName):
	arr = copy.deepcopy(imgFile)
	minValue = round(min(y))
	tmp = [-1] * len(y)
	for i in range(len(y)):
		tmp[int(y[i])] = i 

	for i in range(len(tmp)):
		if tmp[i]== -1:
			k = 0
			for j in range(len(tmp) - i):
				if tmp[i+j]== -1:
					k += 1
				else:
					for m in range(k):
						tmp[i+m] = tmp[i+j] + ((tmp[i+k] - tmp[i])/k * m) 
					break
	if len(arr.shape) == 2:
		for i in range(arr.shape[0]):
			for j in range(arr.shape[1]):
				arr[i,j] = tmp[int(arr[i,j])]
	else:	
		for i in range(imgFile.shape[0]):
			for j in range(imgFile.shape[1]):
				k = arr[i,j][0]
				arr[i,j][0] = tmp[k]
				arr[i,j][1] = tmp[k]
				arr[i,j][2] = tmp[k]
	saveAs(arr, "ree/add_"+imgName.split('/')[-1])
	plt.subplot(1,2,1)
	plt.imshow(arr, cmap = "gray")
	plt.title("Additioned")
	plt.subplot(1,2,2)
	plt.imshow(imgFile, cmap = "gray")
	plt.title("Original")
	plt.show()
	return arr

def equalizer(imgFile,y,imgName):
	arr = copy.deepcopy(imgFile)
	if len(arr.shape) == 2:
		for i in range(arr.shape[0]):
			for j in range(arr.shape[1]):
				arr[i,j] = y[int(arr[i,j])]
	else:
		for i in range(arr.shape[0]):
			for j in range(arr.shape[1]):
				tmp = arr[i,j][0]
				arr[i,j][0] = y[tmp]
				arr[i,j][1] = y[tmp]
				arr[i,j][2] = y[tmp]
	saveAs(arr, "ree/Equal_"+imgName.split('/')[-1])
	plt.subplot(1,2,1)
	plt.imshow(arr, cmap = "gray")
	plt.title("Equalizered")
	plt.subplot(1,2,2)
	plt.imshow(imgFile, cmap = "gray")
	plt.title("Original")
	plt.show()
	return arr

def histo(imgFile, imgName):
	if len(imgFile.shape) == 2:
		height, width = imgFile.shape
		deep = 1
	else:
		height, width, deep = imgFile.shape
	minP = float('inf')
	maxP = float('-inf')
	arr = copy.deepcopy(imgFile)
	for i in range(height):
		for j in range(width):
			tmp = imgFile[i][j]
			if deep > 1:
				tmp = int(sum(tmp)/deep)
			if  tmp > maxP:
				maxP = int(tmp)
			elif tmp < minP:
				minP = int(tmp)
	mx = maxP
	mn = minP
	interval = 256
	x = [i for i in range(interval)]
	step = (mx - mn) / interval
	y = [0] * 256
	ySec = [0] * 256
	if deep > 1:
		for i in range(imgFile.shape[0]):
				for j in range(imgFile.shape[1]):
					y[int(imgFile[i, j][0])] += 1
	else:
		for i in range(imgFile.shape[0]):
				for j in range(imgFile.shape[1]):
					y[int(imgFile[i, j])] += 1
	varTmp = 0
	for i in range(len(y)):
		varTmp += y[i]
		ySec[i] = varTmp
	plt.subplot(1,2,1)
	plt.bar(x,y, width = step, edgecolor='white', linewidth=0.5)
	plt.ylabel("Count")
	plt.title("Гистограмма")

	ySec1 = [0] * 256
	val = imgFile.shape[0] * imgFile.shape[1] 
	for i in range(len(ySec)):
		ySec1[i] = ySec[i] / val * 255
	plt.subplot(1,2,2)
	plt.plot(x,ySec1, linewidth=0.5)
	plt.plot(ySec1, x, linewidth=0.5)
	plt.title("Resizined")
	plt.show()
	#print(ySec1, x)
	t = equalizer(imgFile,ySec1,imgName)
	addition(imgFile,ySec1,imgName)
	return t

def readXCR(fileName,h ,w):
	file = np.fromfile(fileName, dtype = np.ushort).reshape((h, w))
	#tmp = [0x000000] * (file.shape[0] * file.shape[1])
	#tmp = np.reshape(tmp, (300,400))

	test = np.fft.rfft(file[1])
	for i in range(0, file.shape[0], 25):
		test = np.fft.rfft(file[i])
		plt.subplot(6,2,int(i / 25)+1)
		plt.plot(np.fft.rfftfreq(file.shape[1], 0.002), np.abs(test)/file.shape[1],linewidth = 0.6)
	plt.show()
	arr = [0xfff] * (file.shape[0] * file.shape[1])
	arr = np.reshape(arr, (h,w))
	#a = 0x00
	#b = 0x00000000
	pl1 = model.Model(w*h)
	pl1.arr['baseArr'] = np.reshape(pl1.arr['baseArr'], (h,w))
	for i in range(file.shape[0]):
		for j in range(file.shape[1]):
			#a = file[i,j] & 0x0ff
			#b = file[i,j] >> 8
			#tmp[i,j] = b << 24 | a << 16 | a <<8 | a
			arr[i,j] = arr[i,j] & file[i, j]
			pl1.arr['baseArr'][i,j] = arr[i,j]
			#print(i, j, hex(file[i,j]), hex(a), hex(b), hex(tmp[i,j]))
	filterBSF = model.Model(4)
	filterBSF.lpf(1,8,0.002)
	x = [i for i in range(33)]
	ana = analysis.Analysis(filterBSF)
	ana.furie()

	Fpl = []
	pl = []
	an = []
	x = [i for i in range(w)]
	
	for i in range(h):
		pl.append(model.Model(w))
		pl[i].arr['baseArr'] = pl1.arr['baseArr'][i]
		########an.append(analysis.Analysis(pl[i]))
		########if (i % 30) == 0:
		########	an[i].furie()
		########	#plt.subplot(10,1,int(i / 30)+1)
		########	#plt.plot(x, pl[i].arr['C'],linewidth = 0.6)
		########	#plt.grid(True)
		########	
		########	print("Working ", int(i / 3), "%")
	#plt.plot(x, filterBSF.arr['baseArr'],linewidth = 0.6)
	#plt.show()
	
	for i in range(h):
		Fpl.append(model.Model(w))
		Fpl[i].convolution(pl[i],filterBSF)
		Fpl[i].arr['baseArr'] = Fpl[i].arr['baseArr'][5:w+5]
		if (i % 30) == 0:
			print("Working ", int(i / 3), "%")
	print('Successful.')
	lastStep = []
	for i in range(h):
		lastStep.append(Fpl[i].arr["baseArr"])
	pl1.arr['baseArr'] = lastStep
	

	#	Find dependense
	#x = [i for i in range(w-1)]
	#for j in range(int(pl1.arr['baseArr'].shape[0]/30)-5):
	#	an[j*30].autocorrelation(pl[j*30 + 30])
	#	plt.subplot(int(pl1.arr['baseArr'].shape[0]/30),1,j+1)
	#	plt.plot(x, pl[j*30].arr['Rxy'],linewidth = 0.6)
	#	plt.grid(True)
	#
	test = np.fft.rfft(pl1.arr['baseArr'][1])
	print(len(test))
	for i in range(0, len(pl1.arr['baseArr']), 25):
		test = np.fft.rfft(pl1.arr['baseArr'][i])
		plt.subplot(6,2,int(i / 25)+1)
		plt.plot(np.fft.rfftfreq(file.shape[1], 0.002), np.abs(test)/file.shape[1],linewidth = 0.6)
	plt.show()

	plt.subplot(1,2,1)		
	plt.imshow(file, cmap='gray')
	plt.subplot(1,2,2)		
	plt.imshow(pl1.arr['baseArr'], cmap='gray')
	plt.show()
	return "fixed_"+fileName.split('/')[-1], pl1.arr['baseArr']

def showXCR(fileName):
	#file = np.fromfile(fileName, dtype = np.ushort).reshape((h, w))
	#tmp = [0x000000] * (file.shape[0] * file.shape[1])
	#tmp = np.reshape(tmp, (300,400))
	Arr = (0 for i in range(400) for j in range(300))
	Arr = imread("stuff/"+fileName+".jpg", mode="RGB")
	#a = 0x00
	#b = 0x00000000
	pl1 = model.Model(300*400)
	pl1.arr['baseArr'] = np.reshape(pl1.arr['baseArr'], (400,300))
	for i in range(400):
		for j in range(300):
			#a = file[i,j] & 0x0ff
			#b = file[i,j] >> 8
			#tmp[i,j] = b << 24 | a << 16 | a <<8 | a
			#arr[i,j] = arr[i,j] & file[i, j]
			pl1.arr['baseArr'][i,j] = 0xff0f
			#print(i, j, hex(file[i,j]), hex(a), hex(b), hex(tmp[i,j]))
	plt.imshow(Arr, cmap='gray')
	plt.show()

def rndNoice(imgName, percent, isShow = True):
	im = imread(imgName, mode="RGB")
	arr = copy.deepcopy(im)
	#arr = np.array([[0]*3 for i in range(im.shape[0]) for j in range(im.shape[1])])
	#arr = arr.reshape(im.shape[0],im.shape[1],im.shape[2])
	toFix = int(im.shape[0] * im.shape[1] * (percent/100))
	for i in range(toFix):
		x = int(random.random() * (im.shape[0]))
		y = int(random.random() * (im.shape[1]))
		tmp = int(random.random() * (256 + 1))
		arr[x,y][0] = tmp
		arr[x,y][1] = tmp
		arr[x,y][2] = tmp
	if(isShow):
		plt.subplot(1,2,1)
		plt.imshow(im)
		plt.title("Image")
		plt.subplot(1,2,2)
		plt.imshow(arr)
		plt.title("Random noise")
		plt.show()
	else:
		return arr


def saltPeper(imgName, percent, isShow = True):
	im = imread(imgName, mode="RGB")
	arr = copy.deepcopy(im)
	#arr = np.array([[0]*3 for i in range(im.shape[0]) for j in range(im.shape[1])])
	#arr = arr.reshape(im.shape[0],im.shape[1],im.shape[2])
	toFix = int(im.shape[1] * (percent/100))
	for j in range(im.shape[0]):
		for i in range(toFix):
			x = j
			y = int(random.random() * (im.shape[1]))
			tmp = round(random.random()) * 255
			arr[x,y][0] = tmp
			arr[x,y][1] = tmp
			arr[x,y][2] = tmp
	if(isShow):
		plt.subplot(1,2,1)
		plt.imshow(im)
		plt.title("Image")
		plt.subplot(1,2,2)
		plt.imshow(arr)
		plt.title("Salt and peper")
		plt.show()
	else:
		return arr

def averagingFilter(imgName, window, isShow = True):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	arr = copy.deepcopy(im)
	edgex = floor(window / 2)
	edgey = floor(window / 2)
	for x in range(edgex, im.shape[0] - edgex): 
		for y in range(edgey, im.shape[1] - edgey):
			colorMesh = 0
			for fx in range(window):
				for fy in range(window):
					colorMesh += im[x + fx - edgex][y + fy - edgey][0]
			colorMesh = round(colorMesh / (window*window))
			arr[x][y] = colorMesh
	if(isShow):
		plt.subplot(1,2,1)
		plt.imshow(im)
		plt.title("Image")
		plt.subplot(1,2,2)
		plt.imshow(arr)
		plt.title("Median filter")
		plt.show()
	else:
		return arr

def medianFilter(imgName, window, isShow = True):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	arr = copy.deepcopy(im)
	edgex = floor(window / 2)
	edgey = floor(window / 2)
	for x in range(edgex, im.shape[0] - edgex): 
		for y in range(edgey, im.shape[1] - edgey):
			colorMesh = np.array([[0]*3 for i in range(window) for j in range(window)])
			colorMesh = colorMesh.reshape(window,window,3)
			for fx in range(window):
				for fy in range(window):
					colorMesh[fx][fy] = im[x + fx - edgex][y + fy - edgey]
			colorMesh = colorMesh.reshape(1,window*window*3)
			colorMesh = np.sort(colorMesh, axis=None)
			colorMesh = colorMesh.reshape(window,window,3)
			arr[x][y] = colorMesh[floor(window  / 2)][floor(window  / 2)]
	if(isShow):
		plt.subplot(1,2,1)
		plt.imshow(im)
		plt.title("Image")
		plt.subplot(1,2,2)
		plt.imshow(arr)
		plt.title("Median filter")
		plt.show()
	else:
		return arr

def readUShortInList(filename="",n_rows=0, n_cols=0):
	with open(filename, "rb") as file:
		name = file.read()
		dt = np.dtype(('<f', (n_rows, n_cols))) # (n_rows, n_cols,4)))
		out = np.frombuffer(name[:n_rows*n_cols*4], dt)
		return out[0]


def antiSmear(imgName, alfa = 1):
	im = model.Model(src = imgName)
	im.readFile(0)
	img = im.arr['baseArr'].reshape(221,307)

	kernal = model.Model(src = 'Used/kernD76_f4.dat')
	kernal.readFile(0)
	for i in range(307 - kernal.arr['baseArr'].shape[0]):
		kernal.arr['baseArr'] = np.append(kernal.arr['baseArr'],0)
	kernal = kernal.arr['baseArr'].reshape(307)
	
	spectr = np.array([np.fft.fft(img[i]) for i in range(img.shape[0])])

	plt.imshow(np.abs(spectr), cmap="gray")
	plt.show()

	H = np.fft.fft(kernal)
	F_new=np.zeros((spectr.shape[0],spectr.shape[1]),complex)
	for i in range(spectr.shape[0]):
		for j in range(spectr.shape[1]):
			F_new[i,j]=spectr[i,j]*(H[j].conjugate())/(np.abs(H[j])**2+alfa)
	invFu = [ np.fft.ifft(F_new[i]) for i in range(F_new.shape[0])]

	plt.subplot(1,2,1)
	plt.title('Fixed')
	plt.imshow(np.abs(invFu), cmap="gray")
	plt.subplot(1,2,2)
	plt.title('Original')
	plt.imshow(img, cmap="gray")
	plt.show()

def lpf(fcut,m,dt):
	d=[0.35577019,0.24369830,0.07211497,0.00630165]
	arg=2*fcut*dt
	lpw=[]
	lpw.append(arg)
	arg*=np.pi
	for i in range(1, m+1):
		lpw.append(np.sin(arg*i)/(np.pi*i))
	lpw[m]/=2
	sumg=lpw[0]
	for i in range(1, m+1):
		summ=d[0]
		arg=np.pi*i/m
		for k in range(1, 4):
			summ+=2*d[k]*np.cos(arg*k)
		lpw[i]*=summ
		sumg+=2*lpw[i]
	for i in range(0, m):
		lpw[i]/=sumg
	lpwst=[]
	for i in range(m, 0, -1):
		lpwst.append(lpw[i])
	for i in range(0, m+1):
		lpwst.append(lpw[i])
	return lpwst

def hpf(fcut,m,dt):
	hpw=[0]*int(2*m+1)
	lpww=lpf(fcut,m,dt)
	for i in range(0,2*m+1):
		if i==m:
			hpw[i]=1-lpww[i]
		else:
			hpw[i]=-lpww[i]
	return hpw

def convol(arr1,arr2,N=1000,M=1000):
	out=[]
	for n in range(0, N+M):
		arr=0
		m=M-1
		while m>=0:
			if n-m>=0 and n-m<N:
				arr+=arr1[n-m]*arr2[m]
			m-=1
		out.append(arr)
	return out[int(M/2):N+int(M/2)]


def lpfCirculus(imgName):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	im = toGray(im)
	arr = copy.deepcopy(im)
	w=im.shape[0]
	h=im.shape[1]
	lpfrow=lpf(fcut=5*0.5/400,m=8,dt=1)
	lpfcol=lpf(fcut=5*0.5/300,m=8,dt=1)
	out=np.zeros((w,h),float)
	for row in range(w):
		out[row]=convol(arr[row],lpfrow,N=h,M=len(lpfrow))
	out=out.transpose()
	arr=arr.transpose()

	for col in range(h):
		out[col]=convol(out[col],lpfcol,N=w,M=len(lpfcol))
	out=out.transpose()
	arr=arr.transpose()

	out=arr-out
	p=(np.max(out)+np.min(out))/2
	lpfArr = copy.deepcopy(out)
	for i in range(lpfArr.shape[0]):
		for j in range(lpfArr.shape[1]):
			if lpfArr[i,j]>=p:
				lpfArr[i,j]=0
			else:
				lpfArr[i,j]=255
	plt.subplot(1,2,1)
	plt.title('LPF contur with porog')			
	plt.imshow(lpfArr, cmap="gray")
	plt.subplot(1,2,2)
	plt.title('LPF contur')
	plt.imshow(out, cmap="gray")
	plt.show()


def hpfCirculus(imgName):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	im = toGray(im)
	arr = copy.deepcopy(im)
	w=im.shape[0]
	h=im.shape[1]
	hpfrow=hpf(fcut=.9*0.5/400,m=16,dt=1)
	hpfcol=hpf(fcut=.9*0.5/300,m=16,dt=1)
	out=np.zeros((w,h),float)
	for row in range(w):
		out[row]=convol(arr[row],hpfrow,N=h,M=len(hpfrow))
	out=out.transpose()
	arr=arr.transpose()

	for col in range(h):
		out[col]=convol(out[col],hpfcol,N=w,M=len(hpfcol))
	out=out.transpose()
	arr=arr.transpose()

	#out=arr-out
	p=(np.max(out)+np.min(out))/2
	hpfArr = copy.deepcopy(out)
	for i in range(hpfArr.shape[0]):
		for j in range(hpfArr.shape[1]):
			if hpfArr[i,j]>=p:
				hpfArr[i,j]=0
			else:
				hpfArr[i,j]=255
	plt.subplot(1,2,1)
	plt.title('HPF contur with porog')			
	plt.imshow(hpfArr, cmap="gray")
	plt.subplot(1,2,2)
	plt.title('HPF contur')
	plt.imshow(out, cmap="gray")
	plt.show()


def laplasianFilter(imgName, window = 3, isShow = True):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
		img = mod.imread(imgName)
	else:
		im = imgName
		img = imgName
	arr = copy.deepcopy(img)
	edgex = floor(window / 2)
	edgey = floor(window / 2)
	colorMesh = np.array([-1 for i in range(window) for j in range(window)])
	colorMesh = colorMesh.reshape(window,window)
	colorMesh[edgex][edgey] = 8
	 
	img = mod.Laplacian(img, mod.CV_64F) 
	for x in range(edgex, img.shape[0] - edgex): 
		for y in range(edgey, img.shape[1] - edgey):
			break
			for fx in range(window):
				for fy in range(window):
					arr[x][y][0] = im[x + fx - edgex][y + fy - edgey][0] * colorMesh[fx][fy]
					arr[x][y][1] = im[x + fx - edgex][y + fy - edgey][0] * colorMesh[fx][fy]
					arr[x][y][2] = im[x + fx - edgex][y + fy - edgey][0] * colorMesh[fx][fy]

	if(isShow):
		plt.subplot(1,2,1)
		plt.imshow(im, cmap="gray")
		plt.title("Image")
		plt.subplot(1,2,2)
		plt.imshow(img, cmap="gray")
		plt.title("Laplasian filter")
		plt.show()
	return img

def sobelMask(imgName, isShow = True):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	im = toGray(im)

	dx = ndimage.sobel(im, 1)  
	dy = ndimage.sobel(im, 0)  
	mag = np.hypot(dx, dy) 
	 
	mag *= 255.0 / np.max(mag)
	if isShow:
		plt.title('Sobel')  
		plt.imshow(mag, cmap="gray")
		plt.show()
	return mag


def delotation(imgName, window, isShow = True):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	window = 3
	arr = copy.deepcopy(im)
	edgex = floor(window / 2)
	edgey = floor(window / 2)
	for x in range(edgex, im.shape[0] - edgex): 
		for y in range(edgey, im.shape[1] - edgey):
			colorMesh = np.array([0 for i in range(window) for j in range(window)])
			colorMesh = colorMesh.reshape(window,window)
			for fx in range(window):
				for fy in range(window):
					colorMesh[fx][fy] = im[x + fx - edgex][y + fy - edgey][0]
			arr[x][y][0] = np.amax(colorMesh)
			arr[x][y][1] = np.amax(colorMesh)
			arr[x][y][2] = np.amax(colorMesh)
	if(isShow):
		plt.subplot(1,2,1)
		plt.imshow(im)
		plt.title("Image")
		plt.subplot(1,2,2)
		plt.imshow(arr)
		plt.title("Delotation filter")
		plt.show()
	return arr


def erozia(imgName, window, isShow = True):
	if type(window) is int:
		edgex = floor(window / 2)
		edgey = floor(window / 2)
		k = window
	else:
		edgex = floor(window.shape[0] / 2)
		edgey = floor(window.shape[1] / 2)
		k = window.shape[0]
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	arr = copy.deepcopy(im)
	for x in range(edgex, im.shape[0] - edgex): 
		for y in range(edgey, im.shape[1] - edgey):
			colorMesh = np.array([0 for i in range(k) for j in range(k)])
			colorMesh = colorMesh.reshape(k,k)
			for fx in range(k):
				for fy in range(k):
					if type(window) is int:
						colorMesh[fx][fy] = im[x + fx - edgex][y + fy - edgey][0]
					else:
						colorMesh[fx][fy] = im[x + fx - edgex][y + fy - edgey][0] * window[fx - edgex][fy - edgey]
			arr[x][y][0] = np.amin(colorMesh)
			arr[x][y][1] = np.amin(colorMesh)
			arr[x][y][2] = np.amin(colorMesh)
	if(isShow):
		plt.subplot(1,2,1)
		plt.imshow(im)
		plt.title("Image")
		plt.subplot(1,2,2)
		plt.imshow(arr)
		plt.title("Erozia filter")
		plt.show()
	return arr


def drawCircle(arr, x, y):
	arr[x,-y ] = 1 
	arr[y,-x ] = 1 
	arr[y,x  ] = 1 
	arr[x,y  ] = 1 
	arr[-x,y ] = 1
	arr[-y,x ] = 1 
	arr[-y,-x] = 1 
	arr[-x,-y] = 1 



def getRing(D):
	R = int(D/2)
	arr = np.array([[0] for i in range(D+1) for j in range(D+1)])
	arr = arr.reshape(D+1, D+1) 
	switch = 3 - (2 * R)
	x = 0
	y = R
	while x <= y:
		arr[R + x,R - y] = 1 
		arr[R + y,R - x] = 1 
		arr[R + y,R + x] = 1 
		arr[R + x,R + y] = 1 
		arr[R - x,R + y] = 1
		arr[R - y,R + x] = 1 
		arr[R - y,R - x] = 1 
		arr[R - x,R - y] = 1 
		if switch < 0:
			switch = switch + (4 * x) + 6
		else:
			switch = switch + (4 * (x - y)) + 10
			y = y - 1
		x = x + 1
	tt = copy.deepcopy(arr)
	for i in range(tt.shape[0]):
		for j in range(tt.shape[1]):
			if (i - R)**2 + (j - R)**2 <= R**2:
				tt[i,j] = 1
	return tt, arr




####a,b = openIm('Used/grace.jpg') #showInfo saveAs
####a = toGray(a)
####showInfo(a,b)
####a = normize(a)
####for i in range(a.shape[0]):
####	a[i, int(a.shape[1]/2)] = 0
####saveAs(a,'k/changed_grace.jpg')
####negativ(a,b)
####
####
####a,b = openIm('Used/HollywoodLC.jpg')
####a = toGray(a)
####a = hammaCorrect(a, 3.7)
####saveAs(a,'k/hamma_Hollywood.jpg')
####
####
####a,b = openIm('Used/HollywoodLC.jpg')
####a = toGray(a)
####a = hammaCorrect(a, 0.3)
####saveAs(a,'k/hamma_Hollywood.jpg')
####
####
####a,b = openIm('Used/HollywoodLC.jpg')
####a = toGray(a)
####a = logorFunc(a, 3)
####saveAs(a,'k/log_Hollywood.jpg')
####
####
####a,b = openIm('Used/HollywoodLC.jpg')
####a = toGray(a)
####a = logorFunc(a, 1.25)
####saveAs(a,'k/log_Hollywood.jpg')
####
####
####a,b = openIm('Used/grace.jpg')
####a = toGray(a)
####a = resize(a, b, 2)
####saveAs(a,'lineral_'+ b.split('/')[-1])
####
####
####a,b = openIm('Used/grace.jpg')
####a = toGray(a)
####a = resize(a, b, 0.8)
####saveAs(a,'lineral_'+ b.split('/')[-1])
####
####
####a,b = openIm('Used/grace.jpg')
####a = toGray(a)
####v = bilinearResize(a, b, 2)
####saveAs(a,'bilineral_'+ b.split('/')[-1])
####
####
####a,b = openIm('Used/grace.jpg')
####a = toGray(a)
####v = bilinearResize(a, b, 0.8)
####saveAs(a,'bilineral_'+ b.split('/')[-1])



####a,b = readXCR('Used/h400x300.xcr',300,400)
####saveAs(a, b)
####showXCR("fixed_h400x300.xcr")
####
####
####a,b = openIm('Used/HollywoodLC.jpg')
####a = toGray(a)
####histo(a, b)
####
####
####a,b = openIm('Used/grace.jpg')
####a = toGray(a)
####histo(a, b)
####
####a,b = openIm('Used/image2.jpg')
#####a = toGray(a)
####histo(a, b)


####sp = saltPeper("Used/MODEL.jpg", 1, False)
####rn = rndNoice("Used/MODEL.jpg", 1, False)
####plt.subplot(3,4,1)
####plt.imshow(sp)
####plt.title("Salt peper")
####plt.subplot(3,4,2)
####plt.imshow(rn)
####plt.title("Random noise")
####plt.subplot(3,4,3)
####plt.imshow(sp)
####plt.title("Salt peper")
####plt.subplot(3,4,4)
####plt.imshow(rn)
####plt.title("Random noise")
####
####plt.subplot(3,4,5)
####plt.imshow(averagingFilter(sp, 3, False))
####plt.title("SP avarage f, 3x3")
####plt.subplot(3,4,6)
####plt.imshow(averagingFilter(rn, 3, False))
####plt.title("RN avarage f, 3x3")
####plt.subplot(3,4,7)
####plt.imshow(medianFilter(sp, 3, False))
####plt.title("SP median f, 3x3")
####plt.subplot(3,4,8)
####plt.imshow(medianFilter(rn, 3, False))
####plt.title("RN median f, 3x3")
####
####plt.subplot(3,4,9)
####plt.imshow(averagingFilter(sp, 5, False))
####plt.title("SP avarage f, 5x5")
####plt.subplot(3,4,10)
####plt.imshow(averagingFilter(rn, 5, False))
####plt.title("RN avarage f, 5x5")
####plt.subplot(3,4,11)
####plt.imshow(medianFilter(sp, 5, False))
####plt.title("SP median f, 5x5")
####plt.subplot(3,4,12)
####plt.imshow(medianFilter(rn, 5, False))
####plt.title("RN median f, 5x5")
####plt.show()



####antiSmear('Used/blur307x221D.dat', 0.05)
####antiSmear('Used/blur307x221D_N.dat', 1)


####lpfCirculus("Used/MODEL.jpg")
####hpfCirculus("Used/MODEL.jpg")
####lpfCirculus(saltPeper("Used/MODEL.jpg", 1, False))
####hpfCirculus(saltPeper("Used/MODEL.jpg", 1, False))
####lpfCirculus(rndNoice("Used/MODEL.jpg", 1, False))
####hpfCirculus(rndNoice("Used/MODEL.jpg", 1, False))


####sp = saltPeper("Used/MODEL.jpg", 1, False)
####rn = rndNoice("Used/MODEL.jpg", 1, False)
####sobelMask("Used/MODEL.jpg")
####sobelMask(sp)
####sobelMask(rn)
####sobelMask(averagingFilter(sp, 3, False))
####sobelMask(medianFilter(sp, 3, False))
####sobelMask(averagingFilter(rn, 3, False))
####sobelMask(medianFilter(rn, 3, False))


####sp = saltPeper("Used/MODEL.jpg", 1, False)
####rn = rndNoice("Used/MODEL.jpg", 1, False)
####laplasianFilter("Used/MODEL.jpg", 3)
####laplasianFilter(sp)
####laplasianFilter(rn)
####laplasianFilter(averagingFilter(sp, 3, False))
####laplasianFilter(medianFilter(sp, 3, False))
####laplasianFilter(averagingFilter(rn, 3, False))
####laplasianFilter(medianFilter(rn, 3, False))


####path = "Used/MODEL.jpg"
######delotation(saltPeper(path, 5, False), 3)
######erozia(saltPeper(path, 5, False), 3)
####
####arr1 = delotation(path,3, False)
####arr2 = erozia(path,3, False)
####arr1 = toGray(arr1)
####arr2 = toGray(arr2)
######arr3 = erozia(path,window = getRing(5))
######arr3 = toGray(arr3)
######plt.imshow(arr3, cmap="gray")
######plt.show()
####for i in range(arr1.shape[0]):
####	for j in range(arr1.shape[1]):
####		arr1[i][j] = arr1[i][j]  - arr2[i][j]
####plt.imshow(arr1, cmap="gray")
####plt.title("Контур")
####plt.show()
















def binaryThreshold(imgName, por = 0, isShow = True):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	w = im.shape[0]
	h = im.shape[1]
	arr = copy.deepcopy(im)
	cnt = [0] * 256
	for i in range(h):
		for j in range(w):
			a = im[i, j][0]
			b = im[i, j][1]
			c = im[i, j][2]
			S = int(a) + int(b) + int(c)
			cnt[int(S/3)] += 1
	x = [i for i in range(256)]
	plt.bar(x,cnt, width = 1, edgecolor='white', linewidth=0.5)
	plt.ylabel("Count")
	plt.title("Гистограмма")
	plt.show()
	factor = max(range(len(cnt)), key=cnt.__getitem__)
	for i in range(h):
		for j in range(w):
			a = arr[i, j][0]
			b = arr[i, j][1]
			c = arr[i, j][2]
			S = int(a) + int(b) + int(c)
			if (S > ((255 + factor) // 2)*1.95):
				arr[i, j][0] = 255
				arr[i, j][1] = 255
				arr[i, j][2] = 255
			else:
				arr[i, j][0] = 0
				arr[i, j][1] = 0
				arr[i, j][2] = 0
	plt.imshow(arr, cmap="gray")
	plt.show()
	for i in range(h):
		for j in range(w):
			if arr[i, j][0] == 0:
				arr[i, j][0] = 255
				arr[i, j][1] = 255
				arr[i, j][2] = 255
			else:
				arr[i, j][0] = 0
				arr[i, j][1] = 0
				arr[i, j][2] = 0
	tt = copy.deepcopy(arr)
	tt = mod.dilate(tt,mod.getStructuringElement(mod.MORPH_ELLIPSE,(8,8)),iterations = 1)
	###gg = mod.getStructuringElement(mod.MORPH_ELLIPSE,(8,8))
	###tt = mod.erode(tt, gg, iterations = 1)
	#if(isShow):
	#	plt.subplot(1,2,1)
	#	plt.imshow(tt, cmap="gray")
	#	plt.title("Erozied")
	#	plt.subplot(1,2,2)
	#	plt.imshow(arr, cmap="gray")
	#	plt.title("Just porog")
	#	plt.show()
	reStore = mod.erode(tt,mod.getStructuringElement(mod.MORPH_ELLIPSE,(8,8)),iterations = 1)
	#plt.imshow(reStore, cmap="gray")
	#plt.show()
	imsave('stuff/stones1.jpg', reStore)

	src = reStore
	#src = mod.imread('stuff/stones1.jpg', mod.IMREAD_GRAYSCALE)
	#src = mod.bitwise_not(src)


	

	params =  mod.SimpleBlobDetector_Params() 

	params.minArea = 66
	params.maxArea = 89 

	detector = mod.SimpleBlobDetector_create(params) 

	# Detect blobs. 
	keypoints = detector.detect(src) 

	# Draw detected blobs as red circles. 
	# mod.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob 
	im_with_keypoints = mod.drawKeypoints(src, keypoints, np.array([]), (255,0,0), mod.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
	print('Во всех направления: ', len(keypoints))
	plt.subplot(1,3,1)
	plt.imshow(tt, cmap="gray")
	plt.subplot(1,3,2)
	plt.imshow(im_with_keypoints, cmap="gray")
	plt.subplot(1,3,3)
	plt.imshow(reStore, cmap="gray")
	plt.show()
	return arr

def find_templ(img, img_tpl):
	h,w = img_tpl.shape

	#img_tpl = mod.normalize(img_tpl, alpha=0, beta=255,  dst = img_tpl, norm_type=mod.NORM_MINMAX, dtype= mod.CV_8U) 
	#img = mod.normalize(img, alpha=0, beta=255,  dst = img, norm_type=mod.NORM_MINMAX, dtype= mod.CV_8U) 
	img_tpl = mod.cvtColor(img_tpl, mod.COLOR_GRAY2BGR)
	
	match_map = mod.matchTemplate( img, img_tpl, mod.TM_CCOEFF_NORMED)

	max_match_map = np.max(match_map) 
	if(max_match_map < 0.71): 
		return []

	a = 0.8 

	match_map = (match_map >= max_match_map * a  ) * match_map  

	match_map_max = maximum_filter(match_map, size=min(w,h) ) 

	match_map = np.where( (match_map==match_map_max), match_map, 0) 


	ii = np.nonzero(match_map)
	rr = tuple(zip(*ii))

	res = [ [ c[1], c[0], w, h ] for c in rr ]
   
	return res


def typesafe_perspectiveTransform(A,B):
	return mod.perspectiveTransform(A.astype(np.float32, copy=False), B)

def draw_frames(img,coord):
	res = img.copy()
	#print(len(coord))
	for c in coord:
		top_left = (c[0],c[1])
		bottom_right = (c[0] + c[2], c[1] + c[3])
		mod.rectangle(res, top_left, bottom_right, color=(255,0,0), thickness=1 )
	return res 

def binaryThreshold2(imgName, por = 0, isShow = True):
	if(type(imgName) is str):
		im = imread(imgName, mode="RGB")
	else:
		im = imgName
	w = im.shape[0]
	h = im.shape[1]
	arr = copy.deepcopy(im)
	cnt = [0] * 256
	for i in range(h):
		for j in range(w):
			a = im[i, j][0]
			b = im[i, j][1]
			c = im[i, j][2]
			S = int(a) + int(b) + int(c)
			cnt[int(S/3)] += 1
	x = [i for i in range(256)]
	factor = max(range(len(cnt)), key=cnt.__getitem__)
	for i in range(h):
		for j in range(w):
			a = arr[i, j][0]
			b = arr[i, j][1]
			c = arr[i, j][2]
			S = int(a) + int(b) + int(c)
			if (S > ((255 + factor) // 2)*1.95):
				arr[i, j][0] = 255
				arr[i, j][1] = 255
				arr[i, j][2] = 255
			else:
				arr[i, j][0] = 0
				arr[i, j][1] = 0
				arr[i, j][2] = 0
	for i in range(h):
		for j in range(w):
			if arr[i, j][0] == 0:
				arr[i, j][0] = 255
				arr[i, j][1] = 255
				arr[i, j][2] = 255
			else:
				arr[i, j][0] = 0
				arr[i, j][1] = 0
				arr[i, j][2] = 0
	templates = []
	x = mod.getStructuringElement(mod.MORPH_ELLIPSE,(10,10))
	nm = mod.getStructuringElement(mod.MORPH_ELLIPSE,(8,8))
	t = np.zeros_like(x)
	for i in range(1,t.shape[0]-1):
		t[4, i] = 255
	templates.append(t)
	t = np.zeros_like(x)
	for i in range(1,t.shape[0]-1):
		t[i, 4] = 255
	templates.append(t)
	t = np.zeros_like(x)
	for i in range(1, t.shape[0]-1):
		t[i, i] = 255
	templates.append(t)
	t = np.zeros_like(x)
	for i in range(1,t.shape[1]-1):
		t[t.shape[1] - 1 - i, i] = 255
	templates.append(t)
	partitionImg = []
	pindex = 1
	avgObj = 0
	for item in templates:
		first = copy.deepcopy(arr)
		first = mod.dilate(first,item,iterations = 1)	
		second = mod.erode(first,item,iterations = 1)
		plt.subplot(1,2,1)
		plt.imshow(second, cmap="gray")
		plt.subplot(1,2,2)
		plt.imshow(item, cmap="gray")
		img_tpl = mod.bitwise_not(item)
		coord = find_templ(second, img_tpl)
		#img_res =  mod.cvtColor(second, mod.COLOR_BGR2GRAY) 
		img_res = draw_frames(second,coord)
		partitionImg.append(img_res)
		avgObj += len(coord)
		#x,y = FindSubImage(item, second)
		#print(x,y)
		#print(len(keypoints))
		#plt.subplot(1,3,1)
		#plt.imshow(tt, cmap="gray")
		#plt.subplot(1,3,2)
		#plt.imshow(im_with_keypoints, cmap="gray")
		#plt.subplot(1,3,3)
		#plt.imshow(reStore, cmap="gray")
		#plt.show()
		plt.show()
	pindex = 1
	ghg = int(avgObj/len(templates) * 1.3)
	for i in partitionImg:
		plt.subplot(2,2,pindex)
		plt.imshow(i, cmap="gray")
		pindex +=1
	print('Хотя бы в одном направлении: ', ghg)
	plt.show()



#############getCircuit('MODEL.jpg', "LPF", 44) # lpf 253



##delotation(saltPeper("MODEL.jpg", 5, False), 3)
##erozia(saltPeper("MODEL.jpg", 5, False), 3)
#
############arr1 = delotation("MODEL.jpg",5)
############arr2 = erozia("MODEL.jpg",5)
############arr3 = erozia("MODEL.jpg",window = getRing(4))
#############
#############laplasianFilter("MODEL.jpg", 3)
############
############for i in range(arr1.shape[0]):
############	for j in range(arr1.shape[1]):
############		tmp = arr1[i][j][0]  - arr2[i][j][0]
############		arr1[i][j][0] =  tmp
############		arr1[i][j][1] =  tmp
############		arr1[i][j][2] =  tmp
############
############
#############m1 = np.amax(arr1)
#############m2 = np.amin(arr1)
#############for i in range(arr1.shape[0]):
#############	for j in range(arr1.shape[1]):
#############		tmp = int((arr1[i][j][0] - m2)/m1 * 255) + m2
#############		arr1[i][j][0] = tmp 
#############		arr1[i][j][1] = tmp 
#############		arr1[i][j][2] = tmp 
############plt.imshow(arr1, cmap="gray")
############plt.title("Контур")
############plt.show()




binaryThreshold('Used/stones.jpg')
#binaryThreshold2('Used/stones.jpg')

'''



pl1 = model.Model(src = 'v1x11.dat')
pl1.readFile(0)
pl2 = model.Model(src = 'v1w12.dat')
pl2.readFile(0)
ana1 = analysis.Analysis(pl1)
ana2 = analysis.Analysis(pl2)
ana1.autocorrelation()
ana2.autocorrelation()
ana1.autocorrelation(pl2)
ana2.autocorrelation(pl1)
ana1.furie()
ana2.furie()



##win = inOut.inOut([pl1, pl2])
##ana1.statistic()
##ana2.statistic()
##print('M = ', ana1.M,ana2.M)
##print('D = ', ana1.D,ana2.D)
##print('СреднОтклон = ', ana1.standartDev,ana2.standartDev)
##print('СреднКвадр = ', ana1.rndSquare,ana2.rndSquare)
##print('СреднКвадратОшибк = ', ana1.rootMeanSquareErr,ana2.rootMeanSquareErr)
##print('Ассиметрия = ', ana1.asymmetry,ana2.asymmetry)
##print('Экцесс = ', ana1.excess,ana2.excess)
##print('СтандарОтклонение = ', ana1.coefAsymm,ana2.coefAsymm)
##print('Куртосис = ', ana1.coefExce,ana2.coefExce)
##win.smartShow(['sS','sS','sS','sS'],['baseArr','Rxx', 'Rxy','C'])#['sS','sS'],['baseArr','C'])
##
##ana1.furie(pl1.analys.CS)
##ana2.furie(pl2.analys.CS)
##win1 = inOut.inOut([pl1, pl2])
##win.smartShow(['sS'],['reC'])
##ana1.furie(pl2.analys.CS)
##ana2.furie(pl1.analys.CS)
##win1 = inOut.inOut([pl1, pl2])
##win1.smartShow(['sS'],['reC'])
##
##pl3 = model.Model(pl1.N + pl2.N)
##pl3.convolution(pl1,pl2)
##
##win1 = inOut.inOut([pl3])
##win1.smartShow()

filt = model.Model(4)
filt.lpf(32,128,0.002)

filt1 = model.Model(4)
filt1.hpf(32,128,0.002)
pl5 = model.Model(filt.N + pl1.N)
pl5.convolution(pl1,filt)
pl6 = model.Model(filt.N + pl2.N)
pl6.convolution(pl1,filt)

pl7 = model.Model(filt.N + pl1.N)
pl7.convolution(pl1,filt1.child)
pl8 = model.Model(filt.N + pl2.N)
pl8.convolution(pl2,filt1.child)

win1 = inOut.inOut([pl5, pl6,pl7,pl8])
win1.smartShow()

#pl2 = model.Model(4)
#pl2.lpf(10,128,0.002)
#pl3 = model.Model(pl1.N + pl2.N)
#pl3.convolution(pl1,pl2)
#
#
#win = inOut.inOut([pl1,pl3])
#win.smartShow()
####
####pl9 = model.Model(N = 1000, S = 20)
####pl9.fillArrRandom()
####pl9.normalization()
####pl9.child.shift(20, 250, 650)
####pl9.child.child.spikes(1, 4.5, 5, 2.5)
####window1 = inOut.inOut([pl9, pl9.child, pl9.child.child,pl9.child.child.child])
####window1.smartShow()
'''