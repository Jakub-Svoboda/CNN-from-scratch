#SFC project - CNN from scratch
#Author: Jakub Svoboda 
#Date: 2019-10-26


import PyQt5
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
import sys, os
import network
import numpy as np
import csv

class DrawArea(QLabel):
	#Encapsulates the display area widget and its functionality
	def __init__(self):
		super().__init__()
		pixmap =  QPixmap()
		self.setPixmap(pixmap)
		self.resize(pixmap.width(),pixmap.height())
		self.update()

class Window(QWidget):
	#Encapsulates the main window and its functionality
	def __init__(self):
		super().__init__()
		self.setFixedSize(700, 300)	#our window is not resizable
		self.setGeometry(00, 00, 700, 300)
		self.initNet()				#load network

		layout = QGridLayout()
		layout.setVerticalSpacing(30)
		self.setLayout(layout)

		self.drawArea = DrawArea()
		layout.addWidget(self.drawArea, 0, 0 , 3, 1)	#add draw area

		self.clear = QPushButton("Select random number")
		layout.addWidget(self.clear, 0, 1, 1, 1)		#add button
		self.clear.clicked.connect(self.onClickClear)	#connect trigger

		self.classify = QPushButton("Classify")
		self.classify.setDisabled(True)
		layout.addWidget(self.classify, 1, 1, 1, 1)		#add button
		self.classify.clicked.connect(self.onClickClassify) #connect trigger

		self.label_1 = QLabel("Predicted class: ")		#add label
		self.label_1.setFont(QtGui.QFont(None, 15, QtGui.QFont.Normal))
		layout.addWidget(self.label_1, 2, 1, 1, 1)
		
		self.loadSet()	#load dataset
		self.show()		#show window

	def loadSet(self):
		#Loads the dataset into network- readable form
		with open("dataset/mnist-tk.inp") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=' ')
			lineCount = 0
			self.dataset = []
			self.labels = []
			for row in csv_reader:		#for each row
				dataInt = []
				for idx, num in enumerate(row):
					if idx == 784:		#end of data line
						break
					dataInt.append(int(num))
				data = np.array(dataInt)
				data = data.reshape((28,28))
				self.dataset.append(data)
				label = np.zeros((10,1))
				label[int(row[786])][0] = 1
				label = label.reshape(10,)	#turns out (10,1) != (10,)
				self.labels.append(label) 		
				lineCount += 1

			self.dataset = np.array(self.dataset)
			self.showSet = self.dataset
			self.dataset = (self.dataset / 255) - 0.5 #normalize to (-0.5,0.5)
			self.labels = np.array(self.labels)
	
	
	def onClickClassify(self):
		result = self.myNet.predict(self.image)		#predict on a single data point
		self.label_1.setText("Predicted class: " + str(result))

	def onClickClear(self):
		self.classify.setDisabled(False)
		self.r = np.random.randint(0, 9999)
		self.image = self.dataset[self.r]
		num = None
		folder = None
		if(self.r < 1000):			#Select the correct folder where the image is stored
			folder = "00000-00999"	
		elif(self.r < 2000):
			folder = "01000-01999"
		elif(self.r < 3000):
			folder = "02000-02999"
		elif(self.r < 4000):
			folder = "03000-03999"
		elif(self.r < 5000):
			folder = "04000-04999"
		elif(self.r < 6000):
			folder = "05000-05999"
		elif(self.r < 7000):
			folder = "06000-06999"
		elif(self.r < 8000):
			folder = "07000-07999"
		elif(self.r < 9000):
			folder = "08000-08999"
		else:
			folder = "09000-09999"

		if self.r < 10:
			num = "0000" + str(self.r) + "-" + str(np.argmax(self.labels[self.r])) + ".gif"	
		elif self.r	< 100:
			num = "000" + str(self.r) + "-" + str(np.argmax(self.labels[self.r])) + ".gif"	
		elif self.r < 1000:
			num = "00" + str(self.r) + "-" + str(np.argmax(self.labels[self.r])) + ".gif"	
		else:
			num = "0" + str(self.r) + "-" + str(np.argmax(self.labels[self.r])) + ".gif"

		path = 	os.path.join("dataset", "t10k", folder, num)	#full path (os specific)

		pixmap =  QPixmap(path)						#display the number 
		pixmap = pixmap.scaledToHeight(280)
		pixmap = pixmap.scaledToWidth(280)
		self.drawArea.setPixmap(pixmap)
		self.drawArea.resize(pixmap.width(),pixmap.height())
		self.drawArea.update()
				

	def initNet(self):
		#loads the saved network from disk
		self.myNet = network.loadModel()


if __name__ == '__main__':
	np.random.seed(42)
	app = QApplication(sys.argv)
	window = Window()
	
	sys.exit(app.exec_())
	