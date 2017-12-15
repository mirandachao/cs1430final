from glob import glob
import os, random

#
# Split files into training and test.
#
# We're going to randomly select some participants.
probParticipant = 0.86
# Then, for each participant, we're going to randomly select some task folders.
probTask = 0.86


# We'll also count the number of files in each set just to make sure it looks about right
nTrainImages = 0
nTestImages = 0

trainFile = "train.txt"
testFile = "test.txt"

trf = open( trainFile, "w")
tef = open( testFile, "w")

directory = '/course/cs1430/datasets/webgazer/framesdataset'
for f in glob('%s/*/' % directory):
    pDir = glob( f + "/*/")
    r = random.uniform(0,1)

    # Whole participant is in test
    if r > probParticipant:
    	print( "Whole participant in test: " + f )
        for f2 in pDir:
            if (len(glob(f2+"/*")) ==0):
                continue
            else:
                tef.write( os.path.normpath(f2) + '\n')
                nTestImages = nTestImages + len(glob( f2 + "/*.png"))
    # Only parts of participant are in test
    else:
        for f2 in pDir:
            if (len(glob(f2+"/*")) ==0):
                continue
            r = random.uniform(0,1)

            # Specific task is in test set
            if r > probTask:
                tef.write( os.path.normpath(f2) + '\n')
                nTestImages = nTestImages + len(glob( f2 + "/*.png"))
            else:
                trf.write( os.path.normpath(f2) + '\n')
                nTrainImages = nTrainImages + len(glob( f2 + "/*.png"))

trf.close()
tef.close()

print( "Num train images: " + str(nTrainImages) + " | Num test images: " + str(nTestImages) )

#next steps:
#1. Now that all directories are in the train and test files, load images
#2. Normalize/resize images ideally to emphasize eyes.
#3. figure out way to train 
