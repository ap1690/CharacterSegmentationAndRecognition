import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
import math
import numpy as np
from typing import Tuple, Union
def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def detect_corners_from_contour(img, cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
    return np.array(approx_corners)


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped
    
for img_name in ["number (60).jpg","1.jpg"]:
    img = cv2.imread(img_name)
    img = cv2.resize(img, (256, 80))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (15,15),0)

    # _, img_blur = cv2.threshold(img_blur, 250, 255, cv2.THRESH_OTSU)
    img_canny = cv2.Canny(img_blur, 70, 200)
    
    contours, _ = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img, cnt, -1, (255, 0, 0), 3) 
    corners = detect_corners_from_contour(img_gray, cnt)
    img_ptf = four_point_transform(img_gray, corners)

    fig = plt.figure(1, (6,3))
    
    ax = plt.subplot(1,3,1)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("original_img")

    ax = plt.subplot(1,3,2)
    ax.imshow(img_canny, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("canny_img")

    ax = plt.subplot(1,3,3)
    ax.imshow(img_ptf, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("perspective_img")

    plt.show()
