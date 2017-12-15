import os
import csv
import cv2
import math
import numpy as np
from network import net

directory = '/course/cs1430/datasets/webgazer/framesdataset'
dataFile = "gazePredictions.csv"

trainFile = "train.txt"
testFile = "test.txt"

imageWidth = 640
imageHeight = 480
leftArrX = []
leftAnsX = []
rightArrX = []
rightAnsX = []
leftArrY = []
leftAnsY = []
rightArrY = []
rightAnsY = []

error = 0.0
count = 0.0
trf = open( trainFile, "r" )
print "Obtaining Training Data..."
for subDir in trf:
	with open( subDir.strip() + "/" + dataFile ) as f:
		V = csv.reader(f, delimiter=',')
		readCSV = csv.reader(f, delimiter=',')
		for row in readCSV:
			#exit()
			# Tobii has been calibrated such that 0,0 is top left and 1,1 is bottom right on the display.
			tobiiLeftEyeGazeX = float( row[2] )
			tobiiLeftEyeGazeY = float( row[3] )
			tobiiRightEyeGazeX = float( row[4] )
			tobiiRightEyeGazeY = float( row[5] )
			webgazerX = float( row[6] )
			webgazerY = float( row[7] )
			clmTracker = row[8:len(row)-1]
			clmTracker = [float(i) for i in clmTracker]
			clmTrackerInt = [int(i) for i in clmTracker]
			leftEyeX = clmTrackerInt[54]
			leftEyeY = clmTrackerInt[55]
			rightEyeX = clmTrackerInt[64]
			rightEyeY = clmTrackerInt[65]

			tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
			tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

			npClm = np.array(clmTracker) 
			#upperLeftEye
			leftX = np.array([])#npClm[46:50]
			leftX = (np.append(leftX, npClm[54]))/imageWidth

			leftY = np.array([])#npClm[46:50]
			leftY = (np.append(leftY, npClm[55]))/imageHeight
			
			#upperRightEye
			rightX = np.array([])#npClm[56:60]
			rightX = (np.append(rightX, npClm[64]))/imageWidth

			rightY = np.array([])#npClm[56:60]
			rightY = (np.append(rightY, npClm[65]))/imageHeight

			leftArrX.append(leftX)
			leftArrY.append(leftY)
			rightArrX.append(rightX)
			rightArrY.append(rightY)
			#print(leftArrX)
			#exit()
			#leftArrX.append(leftEyeX/imageWidth)
			#leftArrY.append(leftEyeY/imageHeight)
			leftAnsX.append(tobiiLeftEyeGazeX)
			leftAnsY.append(tobiiLeftEyeGazeY)
			#rightArrX.append(rightEyeX/imageWidth)
			#rightArrY.append(rightEyeY/imageHeight)
			rightAnsX.append(tobiiRightEyeGazeX)
			rightAnsY.append(tobiiRightEyeGazeY)

			count += 1.0
			error += math.sqrt(math.pow(tobiiEyeGazeX-webgazerX, 2)+ math.pow(tobiiEyeGazeY-webgazerY, 2))


print "Avg error: ", error/count
trf.close()

network = net()
network.train(np.array(leftArrX), np.array(leftAnsX), "leftX")
network.train(np.array(leftArrY), np.array(leftAnsY), "leftY")
network.train(np.array(rightArrX), np.array(rightAnsX), "rightX")
network.train(np.array(rightArrY), np.array(rightAnsY), "rightY")


tef = open( testFile, "r" )

for subDir in tef:
	with open( subDir.strip() + "/" + dataFile ) as f:
		V = csv.reader(f, delimiter=',')
		readCSV = csv.reader(f, delimiter=',')
		for row in readCSV:

			# Tobii has been calibrated such that 0,0 is top left and 1,1 is bottom right on the display.
			tobiiLeftEyeGazeX = float( row[2] )
			tobiiLeftEyeGazeY = float( row[3] )
			tobiiRightEyeGazeX = float( row[4] )
			tobiiRightEyeGazeY = float( row[5] )
			webgazerX = float( row[6] )
			webgazerY = float( row[7] )
			clmTracker = row[8:len(row)-1]
			clmTracker = [float(i) for i in clmTracker]
			clmTrackerInt = [int(i) for i in clmTracker]
			leftEyeX = clmTrackerInt[54]
			leftEyeY = clmTrackerInt[55]
			rightEyeX = clmTrackerInt[64]
			rightEyeY = clmTrackerInt[65]

			tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
			tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

			npClm = np.array(clmTracker) 
			#upperLeftEye
			leftX = np.array([])#npClm[46:50]
			leftX = (np.append(leftX, npClm[54]))/imageWidth

			leftY = np.array([])#npClm[46:50]
			leftY = (np.append(leftY, npClm[55]))/imageHeight
			
			#upperRightEye
			rightX = np.array([])#npClm[56:60]
			rightX = (np.append(rightX, npClm[64]))/imageWidth

			rightY = np.array([])#npClm[56:60]
			rightY = (np.append(rightY, npClm[65]))/imageHeight

			leftArrX.append(leftX)
			leftArrY.append(leftY)
			rightArrX.append(rightX)
			rightArrY.append(rightY)

			#leftArrX.append(leftEyeX/imageWidth)
			#leftArrY.append(leftEyeY/imageHeight)
			leftAnsX.append(tobiiLeftEyeGazeX)
			leftAnsY.append(tobiiLeftEyeGazeY)
			#rightArrX.append(rightEyeX/imageWidth)
			#rightArrY.append(rightEyeY/imageHeight)
			rightAnsX.append(tobiiRightEyeGazeX)
			rightAnsY.append(tobiiRightEyeGazeY)

			count += 1.0
			error += math.sqrt(math.pow(tobiiEyeGazeX-webgazerX, 2)+ math.pow(tobiiEyeGazeY-webgazerY, 2))


leftGuessesX = network.eval(np.array(leftArrX), np.array(leftAnsX), "leftX")
leftGuessesY = network.eval(np.array(leftArrY), np.array(leftAnsY), "leftY")
rightGuessesX = network.eval(np.array(rightArrX), np.array(rightAnsX), "rightX")
rightGuessesY = network.eval(np.array(rightArrY), np.array(rightAnsY), "rightY")

err = 0.0
for i in range(len(leftGuessesX)):
	guessX = (leftGuessesX[i]+rightGuessesX[i])/2
	guessY = (leftGuessesY[i]+rightGuessesY[i])/2
	ansX = (leftAnsX[i]+rightAnsX[i])/2
	ansY = (leftAnsY[i]+rightAnsY[i])/2
	err += math.sqrt(math.pow(guessX-ansX, 2.0)+math.pow(guessY-ansY, 2.0))

err = err/len(leftGuessesX)

print "Final error on test: ", err



#guesses= x.eval(np.array(leftArrX), np.array(leftAnsX), "leftX")
#print(guesses)



#x = net()
#x.set()


