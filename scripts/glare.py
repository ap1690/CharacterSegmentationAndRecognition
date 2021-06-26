import cv2
import numpy as np
import time
import lpdr
import matplotlib.pyplot as plt
clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
img = cv2.imread('number (16).jpg')
t1 = time.time() 
img = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayimg = gray
GLARE_MIN = np.array([0, 0, 50],np.uint8)
GLARE_MAX = np.array([0, 0, 250],np.uint8)
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)
mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
result1 = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA) 
claheCorrecttedFrame = clahefilter.apply(grayimg)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA) 
grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
result2 = cv2.inpaint(img, mask2, 0.1, cv2.INPAINT_TELEA) 
lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
lab_planes1 = cv2.split(lab1)
clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes1[0] = clahe1.apply(lab_planes1[0])
lab1 = cv2.merge(lab_planes1)
clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
cv2.imshow("IMAGE", img)
cv2.imshow("GRAY", gray)
cv2.imshow("HSV", frame_threshed)
cv2.imshow("CLAHE", clahe_bgr)
cv2.imshow("LAB", lab)
cv2.imshow("HSV + INPAINT", result)
cv2.imshow("INPAINT", result1)
cv2.imshow("CLAHE + INPAINT", result2)  
cv2.imshow("HSV + INPAINT + CLAHE   ", clahe_bgr1)
cv2.waitKey(0)

cv2.imwrite("lolita.jpg",result2)
img=np.array(lpdr.LPD("l.jpg").Y)
plt.imshow(img)
plt.show()
