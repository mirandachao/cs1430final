import os
import csv
import cv2
import math
import numpy as np
from network import net, netIm
from PIL import Image
import time

directory = '/course/cs1430/datasets/webgazer/framesdataset'
dataFile = "gazePredictions.csv"

trainFile = "train.txt"
testFile = "test.txt"

eyeWidth = 24
eyeHeight = 2
imageWidth = 640
imageHeight = 480
numbTrainImgs = 69516
numbTestImgs = 27782
leftArr = np.zeros([numbTrainImgs, 24, 24, 1], dtype= np.uint8)
leftAnsX = np.zeros([numbTrainImgs], dtype= np.uint8)
rightArr = np.zeros([numbTrainImgs, 24, 24, 1], dtype= np.uint8)
rightAnsX = np.zeros([numbTrainImgs], dtype= np.uint8)
leftAnsY = np.zeros([numbTrainImgs], dtype= np.uint8)
rightAnsY = np.zeros([numbTrainImgs], dtype= np.uint8)

error = 0.0
count = 0.0
trf = open( trainFile, "r" )
print "Obtaining Training Data..."
imgIndex = 0
x = 999
for subDir in trf:
	if (imgIndex > x):
		x = 2*x
		break
		print("6000 imgs done")
	with open( subDir.strip() + "/" + dataFile ) as f:
		V = csv.reader(f, delimiter=',')
		readCSV = csv.reader(f, delimiter=',')
		gg = 0
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
			im = Image.open(directory + row[0][1:], 'r')
			boxLeft = (leftEyeX-12, leftEyeY-12, leftEyeX+12, leftEyeY+12)
			boxRight = (rightEyeX-12, rightEyeY-12, rightEyeX+12, rightEyeY+12)
			tmpL = im.crop(boxLeft).convert('L')
			tmpR = im.crop(boxRight).convert('L')
			leftEye = np.asarray(tmpL, dtype= np.uint8)
			#print(leftEye.shape)
			#tmpL.save('./pics/' + str(gg) + "testL.png")
			rightEye = np.asarray(tmpR, dtype= np.uint8)
			#tmpR.save('./pics/'+str(gg)+"testR.png")
			#gg += 1
			#if (gg==1000):
			#	break
			#	exit()

			tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
			tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

			leftArr[imgIndex, :, :, :] = np.expand_dims(leftEye, axis=3)
			rightArr[imgIndex, :, :, :] = np.expand_dims(rightEye, axis=3)

			leftAnsX[imgIndex] = tobiiLeftEyeGazeX
			leftAnsY[imgIndex] = tobiiLeftEyeGazeY
			rightAnsX[imgIndex] = tobiiRightEyeGazeX
			rightAnsY[imgIndex] = tobiiRightEyeGazeY
			
			count += 1.0
			error += math.sqrt(math.pow(tobiiEyeGazeX-webgazerX, 2)+ math.pow(tobiiEyeGazeY-webgazerY, 2))
			imgIndex += 1
			if (imgIndex == 1000):
				break


print(leftArr[0:1000])
print "Avg error: ", error/count
trf.close()

network = netIm()
network.train(leftArr, np.array(leftAnsX), "leftX")
network.train(leftArr, np.array(leftAnsY), "leftY")
network.train(rightArr, np.array(rightAnsX), "rightX")
network.train(rightArr, np.array(rightAnsY), "rightY")

leftArr = np.zeros([numbTestImgs, 24, 24, 1], dtype= np.uint8)
leftAnsX = np.zeros([numbTestImgs], dtype= np.uint8)
rightArr = np.zeros([numbTestImgs, 24, 24, 1], dtype= np.uint8)
rightAnsX = np.zeros([numbTestImgs], dtype= np.uint8)
leftAnsY = np.zeros([numbTestImgs], dtype= np.uint8)
rightAnsY = np.zeros([numbTestImgs], dtype= np.uint8)

x=999
tef = open( testFile, "r" )
imgIndex = 0
for subDir in tef:
	if (imgIndex > x):
		x += 6000
		print("6000 imgs done")
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
			im = Image.open(directory + row[0][1:], 'r')
			boxLeft = (leftEyeX-12, leftEyeY-12, leftEyeX+12, leftEyeY+12)
			boxRight = (rightEyeX-12, rightEyeY-12, rightEyeX+12, rightEyeY+12)
			tmpL = im.crop(boxLeft).convert('L')
			tmpR = im.crop(boxRight).convert('L')
			leftEye = np.asarray(tmpL, dtype= np.uint8)
			#print(leftEye.shape)
			#tmpL.save('./pics/' + str(gg) + "testL.png")
			rightEye = np.asarray(tmpR, dtype= np.uint8)
			#tmpR.save('./pics/'+str(gg)+"testR.png")
			#gg += 1
			#if (gg==100):
			#	exit()

			tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
			tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

			leftArr[imgIndex, :, :, :] = np.expand_dims(leftEye, axis=3)
			rightArr[imgIndex, :, :, :] = np.expand_dims(rightEye, axis=3)

			leftAnsX[imgIndex] = tobiiLeftEyeGazeX
			leftAnsY[imgIndex] = tobiiLeftEyeGazeY
			rightAnsX[imgIndex] = tobiiRightEyeGazeX
			rightAnsY[imgIndex] = tobiiRightEyeGazeY
			
			count += 1.0
			error += math.sqrt(math.pow(tobiiEyeGazeX-webgazerX, 2)+ math.pow(tobiiEyeGazeY-webgazerY, 2))
			imgIndex += 1
			if (imgIndex == 1000):
				break


leftGuessesX = network.eval(leftArr, np.array(leftAnsX), "leftX")
leftGuessesY = network.eval(leftArr, np.array(leftAnsY), "leftY")
rightGuessesX = network.eval(rightArr, np.array(rightAnsX), "rightX")
rightGuessesY = network.eval(rightArr, np.array(rightAnsY), "rightY")

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


