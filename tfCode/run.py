import os
import csv
import cv2
import math
import numpy as np
from network import net
from PIL import Image
import time

directory = '/course/cs1430/datasets/webgazer/framesdataset'
dataFile = "gazePredictions.csv"

trainFile = "train.txt"
testFile = "test.txt"

eyeWidth = 24
eyeHeight = 24
imageWidth = 640
imageHeight = 480
numbTrainImgs = 59413
numbTestImgs = 37885
leftArr = np.zeros([numbTrainImgs, eyeHeight, eyeWidth, 1], dtype= np.uint8)
leftPosX = np.zeros([numbTrainImgs])
leftAnsX = np.zeros([numbTrainImgs], dtype= np.uint8)
rightArr = np.zeros([numbTrainImgs, eyeHeight, eyeWidth, 1], dtype= np.uint8)
rightAnsX = np.zeros([numbTrainImgs], dtype= np.uint8)
rightPosX = np.zeros([numbTrainImgs])
leftAnsY = np.zeros([numbTrainImgs], dtype= np.uint8)
leftPosY = np.zeros([numbTrainImgs])
rightAnsY = np.zeros([numbTrainImgs], dtype= np.uint8)
rightPosY = np.zeros([numbTrainImgs])

error = 0.0
trf = open( trainFile, "r" )
print "Obtaining Training Data..."
imgIndex = 0
breakVal = 1000000
for subDir in trf:
#	break
	if (imgIndex == breakVal):
		break
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
			boxLeft = (leftEyeX-eyeWidth/2, leftEyeY-eyeHeight/2, leftEyeX+eyeWidth/2, leftEyeY+eyeHeight/2)
			boxRight = (rightEyeX-eyeWidth/2, rightEyeY-eyeHeight/2, rightEyeX+eyeWidth/2, rightEyeY+eyeHeight/2)
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

			leftPosX[imgIndex] = leftEyeX/imageWidth
			leftPosY[imgIndex] = leftEyeY/imageHeight
			rightPosX[imgIndex] = rightEyeX/imageWidth
			rightPosY[imgIndex] = rightEyeY/imageHeight

			leftAnsX[imgIndex] = tobiiLeftEyeGazeX
			leftAnsY[imgIndex] = tobiiLeftEyeGazeY
			rightAnsX[imgIndex] = tobiiRightEyeGazeX
			rightAnsY[imgIndex] = tobiiRightEyeGazeY

			error += math.sqrt(math.pow(tobiiEyeGazeX-webgazerX, 2)+ math.pow(tobiiEyeGazeY-webgazerY, 2))
			imgIndex += 1
			if (imgIndex == breakVal):
				break


print "Avg train error from webgazer: ", error/numbTrainImgs
trf.close()

network = net()
network.train(leftArr, leftPosX, np.array(leftAnsX), "leftX")
network.train(leftArr, leftPosY, np.array(leftAnsY), "leftY")
network.train(rightArr, rightPosX, np.array(rightAnsX), "rightX")
network.train(rightArr, rightPosY, np.array(rightAnsY), "rightY")


leftArr = np.zeros([numbTestImgs, eyeHeight, eyeWidth, 1], dtype= np.uint8)
leftPosX = np.zeros([numbTestImgs])
leftAnsX = np.zeros([numbTestImgs], dtype= np.uint8)
rightArr = np.zeros([numbTestImgs, eyeHeight, eyeWidth, 1], dtype= np.uint8)
rightAnsX = np.zeros([numbTestImgs], dtype= np.uint8)
rightPosX = np.zeros([numbTestImgs])
leftAnsY = np.zeros([numbTestImgs], dtype= np.uint8)
leftPosY = np.zeros([numbTestImgs])
rightAnsY = np.zeros([numbTestImgs], dtype= np.uint8)
rightPosY = np.zeros([numbTestImgs])

print("Obtaining Test Data...")
tef = open( testFile, "r" )
imgIndex = 0
error = 0.0
for subDir in tef:
        if (imgIndex == breakVal):
                break
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

                        leftPosX[imgIndex] = leftEyeX/imageWidth
                        leftPosY[imgIndex] = leftEyeY/imageHeight
                        rightPosX[imgIndex] = rightEyeX/imageWidth
                        rightPosY[imgIndex] = rightEyeY/imageHeight

			leftAnsX[imgIndex] = tobiiLeftEyeGazeX
			leftAnsY[imgIndex] = tobiiLeftEyeGazeY
			rightAnsX[imgIndex] = tobiiRightEyeGazeX
			rightAnsY[imgIndex] = tobiiRightEyeGazeY

			error += math.sqrt(math.pow((tobiiEyeGazeX-webgazerX)*imageWidth, 2)+ math.pow((tobiiEyeGazeY-webgazerY)*imageHeight, 2))
			imgIndex += 1
			if (imgIndex == breakVal):
				break

print "Avg pixel error for webgazer on test set: ", error/numbTestImgs

leftGuessesX = network.eval(leftArr, leftPosX, np.zeros([numbTestImgs], dtype= np.uint8), "leftX")
leftGuessesY = network.eval(leftArr, leftPosY, np.zeros([numbTestImgs], dtype= np.uint8), "leftY")
rightGuessesX = network.eval(rightArr, rightPosX, np.zeros([numbTestImgs], dtype= np.uint8), "rightX")
rightGuessesY = network.eval(rightArr, rightPosY, np.zeros([numbTestImgs], dtype= np.uint8), "rightY")

err = 0.0
for i in range(len(leftGuessesX)):
	guessX = (leftGuessesX[i]+rightGuessesX[i])/2
	guessY = (leftGuessesY[i]+rightGuessesY[i])/2
	ansX = (leftAnsX[i]+rightAnsX[i])/2
	ansY = (leftAnsY[i]+rightAnsY[i])/2
	err += math.sqrt(math.pow((guessX-ansX)*imageWidth, 2.0)+math.pow((guessY-ansY)*imageHeight, 2.0))

err = err/numbTestImgs

print "Final pixel error on test: ", err



#guesses= x.eval(np.array(leftArrX), np.array(leftAnsX), "leftX")
#print(guesses)



#x = net()
#x.set()


