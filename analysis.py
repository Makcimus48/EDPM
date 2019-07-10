import copy
from math import sin, cos, sqrt, pi

class Analysis:
	def __init__(self, obj):
		self.arr = obj.arr['baseArr']
		self.obj = obj
		self.N = len(obj.arr['baseArr'])
		self.M =0
		self._M =[]
		self.D =0
		self._D =[]
		self.C = [0] * self.N
		self.CS = [0] * self.N
		self.reC = [0] * self.N
		self.standartDev = 0 #сигма
		self.rndSquare = 0 # тризубец в квадрате
		self.rootMeanSquareErr = 0 # Е как электро движущая сила
		self.asymmetry = 0 # мю 3
		self.excess = 0 # мю 4
		self.coefAsymm = 0 # коэф мю3
		self.coefExce = 0 # коэф мю4
		self.station()
		self.Re = None
		self.Im = None
		self.dRe = None
		self.dIm = None

	def station(self, part = 'None'):
		for elem in self.arr:
			self.M += elem
		self.M /= self.N
		for elem in self.arr:
			self.D += (elem - self.M)**2
		self.D /= self.N
		if(part != 'None'):
			arrs = []
			tmpArr = copy.deepcopy(self.arr)
			size = int(len(self.arr) / (part))
			while len(tmpArr)> size:
				pice = tmpArr[:size]
				arrs.append(pice)
				tmpArr = tmpArr[size:]
			arrs.append(tmpArr)
			for row in arrs:
				M = 0
				D = 0
				for elem in row:
					M += elem
				M /= len(row)
				for elem in row:
					D += (elem - M)**2
				D /= len(row)
				self._M.append(M)
				self._D.append(D)
		self.obj.analys = self

	def statistic(self):
		self.standartDev = sqrt(self.D)
		for elem in self.arr:
			self.rndSquare += elem**2
		self.rndSquare /= self.N
		self.rootMeanSquareErr = sqrt(self.rndSquare)
		for elem in self.arr:
			self.asymmetry += (elem - self.M)**3
		self.asymmetry /= self.N
		for elem in self.arr:
			self.excess += (elem - self.M)**4
		self.excess /= self.N
		self.coefAsymm = self.asymmetry/(self.D * self.standartDev)
		self.coefExce = self.excess/(self.D**2)
		self.obj.analys = self

	def autocorrelation(self, second = None):
		self.station()
		self.statistic()
		M1 = self.M
		M2 = self.M
		chek = False
		if(second != None):
			A = Analysis(second)
			A.station()
			A.statistic()
			M2 = second.analys.M
			chek = True
		x = self.arr
		y = self.arr
		if(second != None):
			y = second.arr['baseArr']
		Rn = []
		L = self.obj.N - 1
		for l in range(L):
			part1 = 0
			for k in range(self.obj.N - l -1):
				part1 += (x[k] - M1)*(y[k+l] - M2) 
			part2 = 0
			for k in range(self.obj.N):
				part2 +=  (x[k] - M1)**2
			if(chek):
				self.obj.arr['Rxy'].append(part1 / part2)
				self.obj.title  = 'Корреляция'
			else:
				self.obj.arr['Rxx'].append(part1 / part2)
				self.obj.title  = 'Автокорреляция'
		self.obj.analys = self

	def furie(self, k = None):
		l = k
		if(k == None):
			k = self.arr
		Re = [0] * self.N
		Im = [0] * self.N
		for i in range(len(k)):
			for j in  range(len(k)):
				Re[i] += k[j] * cos(2*pi*i*j / self.N)
				Im[i] += k[j] * sin(2*pi*i*j / self.N)
			Re[i] = Re[i] / self.N
			Im[i] = Im[i] / self.N
			
			if(l == None):
				self.C[i] = sqrt(Re[i]**2 + Im[i]**2)
				self.CS[i] = Re[i] + Im[i]
				self.Re = Re
				self.Im = Im
			else:
				self.reC[i] = Re[i] + Im[i]
				self.dRe = Re
				self.dIm = Im
		if(l == None):
			self.obj.arr['C'] = self.C
			#self.obj.title = 'Спектр'
		else:
			self.obj.arr['reC'] = self.reC
			self.obj.title = 'Восстановленный'
		self.obj.analys = self

