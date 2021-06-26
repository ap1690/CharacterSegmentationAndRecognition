import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_local
from skimage import measure
import imutils

img=cv2.imread("number (3).jpg")
V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 29, offset=15, method="gaussian")
thresh = (V > T).astype("uint8") * 255
thresh = cv2.bitwise_not(thresh)
#plate = imutils.resize(plate, width=400)
#thresh = imutils.resize(thresh, width=400)
labels = measure.label(thresh, background=0)
charCandidates = np.zeros(thresh.shape, dtype="uint8")
print(len(labels))
for label in np.unique(labels):
			# if this is the background label, ignore it
			if label == 0:
				continue
			labelMask = np.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			cnts,hirarchy = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if len(cnts) > 0:
				# grab the largest contour which corresponds to the component in the mask, then
				# grab the bounding box for the contour
				c = max(cnts, key=cv2.contourArea)
				(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

				# compute the aspect ratio, solidity, and height ratio for the component
				aspectRatio = boxW / float(boxH)
				solidity = cv2.contourArea(c) / float(boxW * boxH)
				heightRatio = boxH / float(img.shape[0])

				# determine if the aspect ratio, solidity, and height of the contour pass
				# the rules tests
				keepAspectRatio = aspectRatio < 1.0
				keepSolidity = solidity > 0.15
				keepHeight = heightRatio > 0.4 and heightRatio < 0.95

				# check to see if the component passes all the tests
				if keepAspectRatio and keepSolidity and keepHeight:
					# compute the convex hull of the contour and draw it on the character
					# candidates mask
					hull = cv2.convexHull(c)
					cv2.drawContours(charCandidates, [hull], -1, 255, -1)
					cv2.imshow("Thresh", charCandidates)
					cv2.waitKey(0)

