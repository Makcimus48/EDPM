import matplotlib.pyplot as plt
import numpy as np
import math

class inOut:
	def __init__(self, objs):
		self.objs = objs
		self.names = []
		self.arrs = []
		self.x = []
		for obj in objs:
			self.arrs.append(obj.arr['baseArr'])
			self.names.append(obj.title)
			self.x.append(obj.x)

	def dumpShow(self,time,result):
		y = result
		x = time
		plt.plot(x,y)
		plt.title('Jopa')
		plt.show()



	def multiAutoShow(self):
		if(len(self.x)<5):
			if(len(self.x) != 1):
				cul = 2
			else: cul = 1
		else:
			cul = 3
		if(len(self.x) % cul == 0):
			row = len(self.x) / cul
		else:
			row = len(self.x) / cul + 1
		for i in range(len(self.arrs)):
			x = self.x[i]
			plt.subplot(row,cul,i+1)
			plt.plot(x,self.arrs[i])
			plt.title(self.names[i])
			plt.grid(True)
		plt.subplots_adjust(hspace = 0.2 + cul * 0.1)
		plt.subplots_adjust(wspace=0.35)
		plt.show()

	def simpleShow(self, obj,  plt, tmp):
		y = obj.arr[tmp]
		x = [i * obj.step for i in range(len(y))]
		if(tmp == 'C'):
			x = x[:int(len(y)/2)]
			y = y[:int(len(y)/2)]
		plt.plot(x,y,linewidth = 0.5)
		if tmp == 'baseArr':
			name = 'Функция'
		elif tmp == 'Rxx':
			name = 'Автокорреляция'
		elif tmp == 'Rxy':
			name = 'Взаимная корреляция' 
		elif tmp == 'C':
			name = 'Сектр' 
		elif tmp == 'reC':
			name = 'Восстановленный' 
		else:
			name = obj.title
		if(tmp == 'tmpArr'): name += ' analysis'
		plt.title(name)
		plt.subplots_adjust(hspace=0.4)
		return plt
		#plt.plot(x,y,linewidth = 0.5)
		#plt.title("Test simple")
		#plt.subplots_adjust(hspace=0.4)

	def smartShow(self, maskType = None, arr = None, interval = 100):
		if(arr != None and maskType != None):
			if(len(arr) != len(maskType)):
				arr = ['tmpArr'] * len(maskType)
		if(maskType == None):
			if(len(self.x)<5):
				if(len(self.x) != 1):
					cul = 2
				else: cul = 1
			else:
				cul = 3
			if(len(self.x) % cul == 0):
				row = len(self.x) / cul
			else:
				row = len(self.x) / cul + 1
			for i in range(len(self.arrs)):
				plt.subplot(row,cul,i+1)
				x = np.arange(self.objs[i].lBorder, self.objs[i].N * self.objs[i].step + self.objs[i].lBorder, self.objs[i].step)#N * step + leftBoarder
				plt.plot(x,self.arrs[i],linewidth = 0.6)
				plt.title(self.names[i])
				plt.grid(True)
			plt.subplots_adjust(hspace = 0.2 + cul * 0.1)
			plt.subplots_adjust(wspace=0.35)
			plt.show()
		else:
			if(len(maskType)<5):
				if(len(maskType) != 1):
					cul = 2
				else: cul = 1
			else:
				cul = 3
			if(len(maskType) % cul == 0):
				row = len(maskType) / cul
			else:
				row = len(maskType) / cul + 1
			for obj in self.objs:
				for i, elem in enumerate(maskType):
					plt.subplot(row,cul,i+1)
					if(maskType[i] == 'sS'):
						self.simpleShow(obj, plt, arr[i])
					if(maskType[i] == 'hG'):
						self.showHisto(obj, plt, interval, arr[i])
					if():
						print(3)
					if():
						print(4)
				plt.subplots_adjust(hspace = 0.2 + cul * 0.1)
				plt.subplots_adjust(wspace=0.35)
				plt.show()		
			

	def showInCulum(self):
		cnt = len(self.arrs)
		for i, row in enumerate(self.arrs):
			x =self.x[i]
			plt.subplot(cnt,1,i+1)
			plt.plot(x,row)
			plt.title(self.names[i])
			plt.grid(True)
		plt.subplots_adjust(hspace=0.4)
		plt.show()

	def showHisto(self, obj, plt = None, interval = None, tmp = 'baseArr'):
		obj = obj.arr[tmp]
		mn = min(obj)
		mx = max(obj)
		if(plt == None):
			interval = 100
		step = (mx - mn) / interval
		y = [0] * interval
		for elem in obj:
			k = int((elem - mn)/step)
			y[k if len(y)> k else k-1] += 1
		if(mn >= -1 and mn < 0):
			x = [-1] * interval
		else:
			x = [int(math.ceil(mn))] * interval
		for i, elem in enumerate(x):
			x[i] += i * step + 0.5*step
		if(plt == None):
			return x, y, step, obj
		else:
			plt.bar(x,y, width = step, edgecolor='white', linewidth=0.5)
			plt.xlabel("У")
			plt.ylabel("Count")
			plt.title("Гистограмма")
			return plt


####################################image = Image.open('grace.jpg')
####################################width, height = image.size
####################################print("Width: ", width ,"Height: " , height)
####################################pix = image.load()
####################################gmin = 256
####################################gmax = 0
####################################gavg = 0
####################################culAvg = [0] * width
####################################rowAvg = [0] * height
####################################for i in range(width):
####################################		for j in range(height):
####################################			a = pix[i, j][0]
####################################			gavg += a
####################################			culAvg[i] += a 
####################################			rowAvg[j] += a
####################################			if a < gmin:
####################################				gmin = a
####################################			if a > gmax:
####################################				gmax = a
####################################gavg /= height * width
####################################for i in range(width):
####################################	culAvg[i] /= height
####################################
####################################for i in range(height):
####################################	rowAvg[i] /= width
####################################
####################################print(gavg)
####################################print()
####################################print("Min: ", gmin, "Max: ", gmax)
####################################
####################################culDis = [0] * width
####################################rowDis = [0] * height
####################################for i in range(width):
####################################		for j in range(height):
####################################			culDis[i] += (pix[i, j][0] - culAvg[i])**2
####################################for i in range(width):
####################################		for j in range(height):
####################################			rowDis[j] += (pix[i, j][0] - rowAvg[j])**2
####################################
####################################x = [i for i in range(width)]
####################################plt.subplot(2,2,1)
####################################plt.plot(x,culAvg)
####################################plt.title("Avg in line")
####################################plt.grid(True)
####################################plt.subplot(2,2,2)
####################################plt.plot(x,culDis)
####################################plt.title("D in line")
####################################plt.grid(True)
####################################X = [i for i in range(height)]
####################################plt.subplot(2,2,3)
####################################plt.plot(X,rowAvg)
####################################plt.title("Avg in colum")
####################################plt.grid(True)
####################################plt.subplot(2,2,4)
####################################plt.plot(X,rowDis)
####################################plt.title("D in colum")
####################################plt.grid(True)
####################################plt.subplots_adjust(hspace=0.4)
####################################plt.show()
####################################
####################################
####################################mn = gmin
####################################mx = gmax
####################################interval = 256
####################################
####################################x = [i for i in range(256)]
####################################step = (mx - mn) / interval
####################################y = [0] * 256
####################################for i in range(width):
####################################		for j in range(height):
####################################			y[pix[i, j][0]] += 1
####################################plt.bar(x,y, width = step, edgecolor='white', linewidth=0.5)
####################################plt.xlabel("У")
####################################plt.ylabel("Count")
####################################plt.title("Гистограмма")
####################################plt.show()
####################################



		
