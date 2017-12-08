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
topextent = h;
for i in range(1, h):
	for j in range(1, w):
		if imthresh[i, j] > 0:
			if i < topextent:
				topextent = i
			if j < leftextent:
				leftextent = j
				break

rightextent = 1
bottomextent = 1
for i in range(h-1, 0, -1):
	for j in range(w-1, 0, -1):
		if imthresh[i, j] > 0:
			if i > bottomextent:
				bottomextent = i
			if j > rightextent:
				rightextent = j
				break
print(leftextent)
print(topextent)
print(rightextent)
print(bottomextent)

imgcrop = imthresh[topextent:bottomextent, leftextent:rightextent]
cv2.imshow('imagecrop', imgcrop)

cv2.waitKey(0)
cv2.destroyAllWindows()
