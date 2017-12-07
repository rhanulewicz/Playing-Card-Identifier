import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('../data/input/AC_straight.jpg', cv2.IMREAD_GRAYSCALE)

# First, we'll isolate the card in the scene and scrub the image
ret,imthresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('image', imthresh)

#Find image extents from left, right, top, and bottom
h, w = img.shape[:2]

leftextent = w;
for i in range(1, h):
	for j in range(1, w):
		if(imthresh[i, j] > 0 and j < leftextent):
			leftextent = j
			break
print(leftextent)
cv2.waitKey(0)
cv2.destroyAllWindows()
