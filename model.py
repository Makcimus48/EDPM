import random
import numpy as np
import math
from math import sin, cos, sqrt, log, pi, exp
import time
import copy
import analysis
import inOut
import filter
import wave, struct

class Model:
	def __init__(self, N = None, S = None, leftBoarder = 0, rightBoarder = None, step = 1, arr = [0], src = None):
		self.child = None
		self.parent = None
		self.analys = None
		self.N = N
		self.src = src
		self.S = S
		self.title = None
		self.step = step
		self.arr = {'baseArr': [], 'tmpArr': [], 'Rxx': [], 'Rxy': [] , 'C': [], 'reC': [], 'filtered': [], 'img': []}
		if(src != None):
			self.readFile(leftBoarder)
		else:
			self.arr = {'baseArr': arr * self.N, 'tmpArr': [], 'Rxx': [], 'Rxy': [] , 'C': [], 'reC': [], 'filtered': []}
			self.lBorder = leftBoarder
			self.rBorder = rightBoarder or N * step + leftBoarder
		self.x = np.arange(self.lBorder, self.rBorder, self.step)
		self.currentSeed = int(round(time.time() * 1000))
		self.min = 0
		self.max = 0
		self.time = []
		self._t = 0.002
		self.__minMax()

	def root(self):
		isRoot = self
		while isRoot.parent != None:
			isRoot = isRoot.parent
		return isRoot

	def get(self, i = 0):
		isIt = self.root()
		for i in range(i):
			isIt = isIt.child
			#if(isIt.child != None):
			#	isIt = isIt.child
			#else:
			#	return isIt		
		return isIt

	def last(self):
		isLast = self
		while isLast.child != None:
			isLast = isLast.child
		return isLast

	def __minMax(self):
		for elem in self.arr['baseArr']:
			if elem < self.min:
				self.min = elem
			if elem > self.max:
				self.max = elem
	
	def normalization(self, S):
		m1 = min(self.arr['baseArr'])
		m2 = max(self.arr['baseArr'])
		modifybaseArr = []
		for elem in self.arr['baseArr']:
			modifybaseArr.append(((elem - m1)/(m2 - m1) - 0.5)*2*S) 
		self.normArr = modifybaseArr
		newobj = copy.deepcopy(self)
		#self.child = Model(title = self.title + " N", S = self.S, leftBoarder = self.lBorder, rightBoarder = self.rBorder, step = self.step, arr = modifybaseArr)
		newobj.title = "Normalizy " + self.title
		newobj.arr['baseArr'] = modifybaseArr
		newobj.S = S
		newobj.parent = self
		self.child = newobj

	# Good working func, but tmp alwas 4 byte type float
	def readFile(self, leftBoarder):
		obj = np.fromfile(self.src, '<f4')
		self.arr['baseArr'] = obj
		self.N = len(obj)
		self.lBorder = leftBoarder
		self.rBorder = self.N * self.step + leftBoarder

	def heartSin(self):
		self.harmonic_process(aTmp = [130], fTmp = [7])
		for i in range(self.N):
			self.arr['baseArr'][i] *= math.exp(-7 * i * 0.005)

	def heartSpicks(self):
		for i in range(4):
			self.arr['baseArr'][200*(i+1)] = 1


	def reflectSpicks(self):
			self.arr['baseArr'][467] = 1
			self.arr['baseArr'][668] = -0.8
			#self.arr['baseArr'][289] = 0.27
			#self.arr['baseArr'][300] = 0.15
			#self.arr['baseArr'][312] = 0.08
			#self.arr['baseArr'][341] = -0.8
			#self.arr['baseArr'][352] = -0.47
			#self.arr['baseArr'][363] = -0.21
			#self.arr['baseArr'][374] = -0.1
			#self.arr['baseArr'][386] = -0.04
			self.title = 'Simulated cube border speaks'

	# obj1 - spicks, obj2 - template
	def convolution(self, obj1, obj2):
		self.arr['baseArr'] = [0] * (len(obj1.arr['baseArr']) + len(obj2.arr['baseArr']))
		for i in range(len(obj1.arr['baseArr']) + len(obj2.arr['baseArr'])):
			for j in range(len(obj2.arr['baseArr'])):
				if(i-j < len(obj1.arr['baseArr'])):
					#print(i - j, i, j)
					self.arr['baseArr'][i] += obj1.arr['baseArr'][i-j] * obj2.arr['baseArr'][j] 
		self.N = len(self.arr['baseArr'])
		self.x = np.arange(0,self.N,1)
		if obj2.title == "LPF" or obj2.title == "HPF" or obj2.title == "BPF" or obj2.title == "BSF":
			self.title = obj2.title
		else:
			self.title = 'Свёртка'


	#Good working func, but can optimize
	def harmonic_process(self, _t = 0.002, poly = False, aTmp = None, fTmp = None, deep = None):
		if(deep == None):
			if(aTmp != None or fTmp != None):
				if(aTmp == None): aTmp = []
				if(fTmp == None): fTmp = []
				deep = max(len(aTmp),len(fTmp))
				poly = True
			elif poly:
				deep = int(math.ceil(random.random() * 20))
				aTmp = fTmp = []
			else:
				deep = 1
				aTmp = fTmp = []
			if(deep == 1):
				poly = False
		else:
			if(aTmp == None): aTmp = []
			if(fTmp == None): fTmp = []
			if(deep < max(len(aTmp),len(fTmp))):
				deep = max(len(aTmp),len(fTmp))
			if(deep == 1):
				poly = False
			else:
				poly = True
		dt = _t
		for i in range(deep):
			if(i < len(fTmp)):
				f = fTmp[i]
			else:
				f = int(math.ceil(random.random() * 1050))
			if(i < len(aTmp)):
				a = aTmp[i]
			else:
				a = int(math.ceil(random.random() * 200))
			for i in range(self.N):
				self.arr['baseArr'][i] = a * sin(2 * math.pi * f * dt*i)
		if(poly):
			self.title = "Poly-harmonic process"
		else:
			self.title = "Harmonic process"
		self._t = _t
		self.__minMax()

	def readWAVfile(self):
		waveFile = wave.open(self.src, 'r')
		List = []
		length = waveFile.getnframes()
		for i in range(0,length):
			waveData = waveFile.readframes(1)
			data = struct.unpack("<hh", waveData)
			List.append(int(data[0]))
		self.arr['baseArr'] = List
		self.N = len(List)
		self.rBorder = self.N * self.step + self.lBorder

	def saveWAV(self):
		print(1)

	def trend(self, type):
		if(type == 'x'):
			for i in range(self.N):
				self.arr['baseArr'][i] += i*1.7
				self.title = "Linear increase"
		elif(type == '-x'):
			for i in range(self.N):
				self.arr['baseArr'][i] += 1500 - i*1.5
				self.title = "Linear decrease"
		elif(type == 'e'):
			for j in range(int(-self.N/2), int(self.N/2)):
				part = 1
				y = 1
				k = 0.0185 * j
				for i in range(2,101):
					part *= k / i
					y += part
				self.arr['baseArr'][int(j + self.N/2)] += y
				self.title = "Increasing exponent"
		elif(type == '-e'):
			for j in range(int(-self.N/2), int(self.N/2)):
				part = 1
				y = 1
				k = -0.0185 * j
				for i in range(2,101):
					part *= k / i
					y += part
				self.arr['baseArr'][int(j + self.N/2)] += y
				self.title = "Decreasing exponent"
		elif(type == 'mix'):
			for j in range(int(-self.N/8), int(self.N/8)):
				part = 1
				y = 1
				k = 0.073 * j
				for i in range(2,101):
					part *= k / i
					y += part
				self.arr['baseArr'][int(j + self.N/8)] += y
			for i in range(int(self.N/4)):
				self.arr['baseArr'][int(i + self.N/4)] += 1006 - 2.784*i
			for i in range(int(self.N/4)):
				self.arr['baseArr'][int(i + 2*self.N/4)] += i*1.192 + 313
			for j in range(int(-self.N/8), int(self.N/8)):
				part = 1
				y = 1
				k = -0.0685 * j
				for i in range(2,101):
					part *= k / i
					y += part
				self.arr['baseArr'][int(j + self.N/8 + self.N/4 + self.N/2)] += y
				self.title = "Comprehensive function"
		self.__minMax()

	def fillArrRandom(self):
		for i in range(self.N):
			self.arr['baseArr'][i] += random.random()
		self.title = "Default random"
		self.__minMax()
		print()

	def fillArrSelfRandom(self):
		a = 6364136223846793005
		c = 1442695040888963407
		baseArr = []
		for i in range(self.N):
			num1 = a * self.currentSeed + c
			num2 = a * num1 + c
			num3 = a * num2 + c
			num4 = a * num3 + c
			part1 = (num1 & 0xffff000000000000)<<64
			part2 = num4 & 0x000000000000ffff<<96
			part3 = (num2 & 0x00000000ffff0000)<<64
			part4 = (num3 & 0x0000ffff00000000)<<32
			part5 = (num1 & 0x0000ffff00000000)<<16
			part6 = num2 & 0x000000000000ffff<<32
			part7 = (num4 & 0xffff000000000000)>>32
			part8 = (num3 & 0x00000000ffff0000)>>16
			fullNum = part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8
			elem = float(sqrt(fullNum/c))
			k = sqrt(sqrt(elem))
			# своя но странная
			#makernd = 0.5*sin(elem) + 0.5
			# своя но с приближением к гаусовской кривой
			makernd = ((0.448*((0.5*sin(2*elem) + 0.5) + 0.8*sin(4*elem)) + 0.25) - 0.3*sin(7*elem) + 0.497*cos(2.5*elem) -  0.5*sin(1.2*elem))/ 2.9505 + 0.35
			#makernd = fullNum
			#while makernd > 1:
			#	makernd = log(makernd,14)
			self.arr['baseArr'][i] = makernd
			self.currentSeed = int(fullNum)
		self.title = "My random"
		self.__minMax()

	def shift(self, Count, lBord = 0, rBord = -1):
		if rBord == -1: rBord = self.N
		if lBord < 0 or rBord > self.N: 
			lBord = 0 
			rBord = self.N
		newobj = copy.deepcopy(self)
		newobj.title = "Shift"
		for i, elem in enumerate(self.arr['baseArr']):
			if i >= lBord and i <= rBord: newobj.arr['baseArr'][i] += Count
		newobj.parent = self
		self.child = newobj

	def spikes(self, p1, p2, p3, p4):
		spike = []
		rnd = 2 + int(round(p1 * self.N * 0.01 * random.random()))
		if(self.S == None):
			Scale = 100
		else:
			Scale = self.S
		heigth = p3 * Scale 
		k = heigth  * p4
		for i in range(rnd):
			spike.append(int(round(2*(random.random() - 0.5)))*heigth + k * (random.random() - 0.5))
		i = 0
		newobj = copy.deepcopy(self)
		if(len(newobj.arr['baseArr'])!= newobj.N):
			newobj.arr['baseArr'] = [0] * newobj.N
		while len(spike) != 0:
			r = random.random()*100
			if(r >= p2 and r <= 2*random.random()*p2):
				newobj.arr['baseArr'][i] += spike.pop()
			if(i+1 > self.N): 
				i = 0
			else:
				i+=1
		newobj.title = "Spikes"
		newobj.parent = self
		self.child = newobj

	def antiShift(self):
		if(self.analys == None):
			analysis.Analysis(self)
		newobj = copy.deepcopy(self)
		for i in range(len(newobj.arr['baseArr'])):
			newobj.arr['baseArr'][i] -= self.analys.M
		newobj.title = "Anti Shift"
		newobj.parent = self
		self.child = newobj 

	def antiSpikes(self):
		self.S = None
		newobj = copy.deepcopy(self)
		if(self.analys == None):
			analysis.Analysis(self)
		lb = None
		rb = None
		arr = newobj.arr['baseArr']
		if(self.S == None):
			x, y, step, obj = inOut.inOut.showHisto(self, self)
			mn = min(obj)
			mx = max(obj)
			minI = 0
			maxI = 1
			currMinI = None
			currMaxI = None
			zero = False
			for i in range(len(y)):
				if(y[i]!= 0):
					zero = True
					currMinI = i
					for j in range(i+1,len(y)):
						if(y[j] != 0 or zero):
							if(y[j] == 0):
								zero = False
							else:
								zero = True
							currMaxI = j
						else: 
							break
					if(maxI - minI < currMaxI - currMinI):
						minI = currMinI
						maxI = currMaxI
			lb = 1.2 * (mn + minI * step + 0.5*step)
			rb = 1.2 * (mn + maxI * step + 0.5*step)
		else:
			lb = -1.2 * self.S
			rb = 1.2 * self.S
		for i in range(len(arr)):
			if(arr[i]<lb or arr[i]>rb):
				arr[i] = (arr[i-1] + arr[i+1]) / 2
		newobj.arr['baseArr'] = arr
		newobj.title = 'Antispike'
		newobj.parent = self
		self.child = newobj

	# Not finished func
	#Для расширения функцианала sum(rnd i + harm i), где harm i = A0 * sin(2 * Pi * f0 * dt)
	# A0 = 0.005*S, f0 = 5, dt = 0.002
	def antiRandom(self, M = 100):
		newobj = copy.deepcopy(self)
		for i in range(M):
			newobj.fillArrRandom()
		for j in range(len(newobj.arr['baseArr'])):
			newobj.arr['baseArr'][j] /= M
		newobj.title = 'AntiRandom'
		newobj.parent = self
		self.child = newobj

	
	def getTrend(self, K = 10):
		avg = 0
		for i in range(K):
			avg += self.arr['baseArr'][i]
		avg /= K
		newobj = copy.deepcopy(self)
		for j in range(len(newobj.arr['baseArr']) - K):
			newobj.arr['baseArr'][j] = avg
			avg += (newobj.arr['baseArr'][K+j]-newobj.arr['baseArr'][j])/K
		newobj.title = 'GetTrend'
		newobj.parent = self
		self.child = newobj

	def decimation(self, k):
		newobj = copy.deepcopy(self)
		gLen = len(self.arr['baseArr'])
		newobj.arr['baseArr'] = []
		for i in range(gLen):
			if(i % k == 0):
				newobj.arr['baseArr'].append(self.arr['baseArr'][i])
		newobj.N = int(self.N / k)
		newobj.step = self.step * k
		newobj.title = 'Decimated'
		newobj.parent = self
		self.child = newobj

	def inerpolate(self):
		if(self.analys == None):
			analysis.Analysis(self)
		newobj = copy.deepcopy(self)
		gLen = len(self.arr['baseArr'])
		newobj.arr['baseArr'] = []
		for i in range(gLen - 1):
			avg = 0
			for j in range(5):
				avg += self.arr['baseArr'][i]
			avg /= 5
			tmp = (self.arr['baseArr'][i]+self.arr['baseArr'][i+1])/2
			newobj.arr['baseArr'].append(self.arr['baseArr'][i])
			newobj.arr['baseArr'].append(tmp)# + avg - self.arr['baseArr'][i+1])
		newobj.arr['baseArr'].append(self.arr['baseArr'][len(self.arr['baseArr'])-1])
		tmp1 = self.arr['baseArr'][len(self.arr['baseArr'])- 3] - self.arr['baseArr'][len(self.arr['baseArr'])- 2]
		tmp2 = self.arr['baseArr'][len(self.arr['baseArr'])- 2] - self.arr['baseArr'][len(self.arr['baseArr'])- 1]
		tmp = tmp1 - tmp2
		newobj.arr['baseArr'].append(self.arr['baseArr'][len(self.arr['baseArr'])-1] + tmp)
		newobj.N = self.N * 2
		newobj.step = self.step / 2
		newobj.title = 'Inerpolated'
		newobj.parent = self
		self.child = newobj

	def lpf(self, fc, m, dt = 0.002):
		# input:
		#	fc - low cut frequency
		#	m - filter operator length
		#	dt - sampling interval
		# output:
		#	lpw[] = (m+1) - low pass filter weigths
		d = [0.35577019, 0.24136983, 0.07211497, 0.00630165]
		# rectangular part weights
		lpw = [0 for i in range(m + 1)]
		self.N = 2 * m + 1
		arg = 2 * fc * dt
		lpw[0] = arg
		arg *= math.pi;
		for i in range(1,m+1):
			lpw[i] = sin(arg*i)/(math.pi*i)
		# trapezoid smoothing at the end
		lpw[m] /= 2
		# Potter P310 smoothing window
		sumg = lpw[0];
		for i in range(1,m+1):
			sm = d[0]
			arg = math.pi*i/m
			for k in range(1,len(d)):
				sm += 2 * d[k] * cos(arg*k)
			lpw[i] *= sm;
			sumg += 2 * lpw[i]

		# normalizing
		for i in range(m):
			lpw[i] /= sumg
		lpwM = [0] * (2 * m + 1)
		self.time = [i*dt for i in range(len(lpwM))]
		for i in range(len(lpwM)):
			lpwM[i] = lpw[abs(m-i)]
		self.arr['baseArr'] = lpwM
		#inOut.inOut.dumpShow(self, time = self.time, result = self.arr['baseArr'])
		self.title = "LPF"
		self.__minMax()

	def hpf(self, fc, m, dt = 0.002):
		self.lpf(fc, m, dt)
		newobj = copy.deepcopy(self)
		for i in range(2*m +1):
			if(i == m):
				newobj.arr['baseArr'][i] = 1 - self.arr['baseArr'][i]
			else:
				newobj.arr['baseArr'][i] = - self.arr['baseArr'][i]
		newobj.title = 'HPF'
		newobj.parent = self
		self.child = newobj

	def bpf(self, f1, f2, m, dt = 0.002):
		self.lpf(f1, m, dt)
		lpw1 = self.arr['baseArr']
		self.lpf(f2, m, dt)
		lpw2 = self.arr['baseArr']
		newobj = copy.deepcopy(self)
		for i in range(2*m + 1):
			newobj.arr['baseArr'][i] = lpw2[i] - lpw1[i]
		newobj.title = 'BPF'
		newobj.parent = self
		self.child = newobj

	def bsf(self, f1, f2, m, dt = 0.002):
		self.lpf(f1, m, dt)
		lpw1 = self.arr['baseArr']
		self.lpf(f2, m, dt)
		lpw2 = self.arr['baseArr']
		newobj = copy.deepcopy(self)
		for i in range(2*m + 1):
			if(i == m):
				newobj.arr['baseArr'][i] = 1 + lpw1[i] - lpw2[i]
			else:
				newobj.arr['baseArr'][i] = lpw1[i] - lpw2[i]
		newobj.title = 'BSF'
		newobj.parent = self
		self.child = newobj

	def deconvolve(self, g1, f1):
		g = g1.arr['baseArr']
		f = f1.arr['baseArr']
		lenh = len(g) - len(f)
		mtx = [[0 for x in range(lenh+1)] for y in g]
		for hindex in range(lenh):
			for findex, fval in enumerate(f):
				gindex = hindex + findex
				mtx[gindex][hindex] = fval
		for gindex, gval in enumerate(g):		
			mtx[gindex][lenh] = gval
		self.ToReducedRowEchelonForm( mtx )
		self.arr['baseArr'] = [mtx[i][lenh] for i in range(lenh)] 
		self.N = len(self.arr['baseArr'])
		self.title = 'Deconvolution'

	def ToReducedRowEchelonForm(self, M ):
		if not M: return
		lead = 0
		rowCount = len(M)
		columnCount = len(M[0])
		for r in range(rowCount):
			if lead >= columnCount:
				return
			i = r
			while M[i][lead] == 0:
				i += 1
				if i == rowCount:
					i = r
					lead += 1
					if columnCount == lead:
						return
			M[i],M[r] = M[r],M[i]
			lv = M[r][lead]
			M[r] = [ mrx / lv for mrx in M[r]]
			for i in range(rowCount):
				if i != r:
					lv = M[i][lead]
					M[i] = [ iv - lv*rv for rv,iv in zip(M[r],M[i])]
			lead += 1
		return M