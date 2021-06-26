import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import lpdr
for i in glob.glob("*.jpg"):
    print(i)
    try:
        img=np.array(lpdr.LPD(i).Y)
    except:
        cv2.imwrite("blur/"+i,cv2.imread(i))
    i_img=img.copy()
    img= img.astype('float32')*255
    img=img.astype(np.uint8)
    img=cv2.resize(img,(300,100))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=np.expand_dims(gray,axis=2)
    gray_blurr = cv2.GaussianBlur(gray, (15, 15), 0)
    #cv2.imwrite("blur/"+i,gray_blurr)
    #thresh = cv2.adaptiveThreshold(gray_blurr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 1)
    #gray_blur = cv2.GaussianBlur(thresh, (15, 15), 0)
    #kernel = np.ones((3, 3), np.uint8)
    #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    #cont_img = closing.copy()
    #cv2.imshow("d",closing)
    #cv2.waitKey(0)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edged = cv2.Canny(gray_blur, 20, 200)
    #contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thresh = cv2.threshold(gray_blurr, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("x",thresh)
    cv2.waitKey(0)
    break
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(gray.shape, dtype="uint8")
    for j in range(1, numLabels):
        x = stats[j, cv2.CC_STAT_LEFT]
        y = stats[j, cv2.CC_STAT_TOP]
        w = stats[j, cv2.CC_STAT_WIDTH]
        h = stats[j, cv2.CC_STAT_HEIGHT]
        area = stats[j, cv2.CC_STAT_AREA]
        keepWidth = w > 5 and w < 500
        keepHeight = h > 10 and h < 300
        keepArea = area > 300 and area < 5000
        print("[INFO] keeping connected component '{}'".format(j))
        componentMask = (labels == j).astype("uint8") * 255
        mask = cv2.bitwise_or(mask, componentMask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel) 
    cv2.imwrite("mask/"+i,mask)
      
