import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('../data/input/AC_straight_rot_ext2.jpg', cv2.IMREAD_GRAYSCALE)

# First, we'll isolate the card in the scene and scrub the image
ret,imthresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# Perform opening on thresholded image to remove noise in space
kernel = np.ones((8,8), np.uint8)
imthresh_open = cv2.morphologyEx(imthresh, cv2.MORPH_OPEN, kernel)

# Get corner points using rotated rectangle
_, contours, _ = cv2.findContours(imthresh_open,2,1)
cnt = contours[0]
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

cornerNW = box[1]
cornerNE = box[2]
cornerSW = box[0]
cornerSE = box[3]

# Do some homography stuff to "straighten" the card
pts_src = np.array([cornerNW, cornerNE, cornerSW, cornerSE])
pts_dst = np.array([[0.0, 0.0], [691, 0.0], [0.0, 1056],[691, 1056]])
im_dst = np.zeros((1056, 691, 1), np.uint8)
homo, status = cv2.findHomography(pts_src, pts_dst)
im_homo = cv2.warpPerspective(imthresh, homo, (im_dst.shape[1],im_dst.shape[0]))
cv2.imshow('homo', im_homo);

# If im_homo is distorted we need to "stretch" the corners until it's not
	# Look at approxPoly for anothter solution to this

# Find image extents from left, right, top, and bottom
h, w = img.shape[:2]
leftextent = w;
topextent = h;
for i in range(1, h):
	for j in range(1, w):
		if imthresh_open[i, j] > 0:
			if i < topextent:
				topextent = i
			if j < leftextent:
				leftextent = j
				break
rightextent = 1
bottomextent = 1
for i in range(h-1, 0, -1):
	for j in range(w-1, 0, -1):
		if imthresh_open[i, j] > 0:
			if i > bottomextent:
				bottomextent = i
			if j > rightextent:
				rightextent = j
				break

# Crop image to extents
#imgcrop = imthresh[topextent:bottomextent, leftextent:rightextent]
#cv2.imshow('imagecrop', imgcrop)

#Crop image based on corners
#imgcrop2 = imthresh[corner1[0]:corner3[0], corner1[1]:corner2[1]]
#h, w = imgcrop.shape[:2]
#rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)
#img_rotation = cv2.warpAffine(imgcrop, rotation_matrix, (w, h))
#cv2.imshow('rot', imgcrop2)

cv2.waitKey(0)
cv2.destroyAllWindows()
