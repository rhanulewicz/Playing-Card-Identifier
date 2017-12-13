import cv2
import numpy as np
import matplotlib.pyplot as plt 
from os import listdir
from os.path import isfile, join

img = cv2.imread('../data/input/AC_straight_rot_ext.jpg', cv2.IMREAD_GRAYSCALE)

# First, we'll threshold the image and make it a binary image 
ret,imthresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# Perform opening on thresholded image to remove noise in space
kernel = np.ones((8,8), np.uint8)
imthresh_open = cv2.morphologyEx(imthresh, cv2.MORPH_OPEN, kernel)

# Get 4 corner points using rotated rectangle
_, contours, _ = cv2.findContours(imthresh_open,2,1)
cnt = contours[0]
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

cornerNW = box[1]
cornerNE = box[2]
cornerSW = box[0]
cornerSE = box[3]

## TODO
# To get 4 corners of a distorted image, we can use convex hull
hull = cv2.convexHull(cnt)
# Get furthest points of the hull in the 8 cardinal directions
# (Not sure the following is correct, should be verified)
# N: 	Maximize -y
# NE: 	Maximize x - y
# E: 	Maximize x
# SE: 	Maximize x + y
# S: 	Maximize y
# SW:	Maximize -x + y
# W: 	Maximize -x
# NW:	Maximize -x - y
# The resulting hull will be an octagon
# Take the 4 longest sides of and intersect them to find the 4 corners

# Another solution might involve approxPoly but I don't know how to use it
##

# Do some homography stuff to "straighten" the card
pts_src = np.array([cornerNW, cornerNE, cornerSW, cornerSE])
pts_dst = np.array([[0.0, 0.0], [691, 0.0], [0.0, 1056],[691, 1056]])
im_dst = np.zeros((1056, 691, 1), np.uint8)
homo, status = cv2.findHomography(pts_src, pts_dst)
im_homo = cv2.warpPerspective(imthresh, homo, (im_dst.shape[1],im_dst.shape[0]))
cv2.imshow('homo', im_homo);

# Isolate the card letter and symbol
im_sym = im_homo[18:300, 18:150]
ret2,im_sym_t = cv2.threshold(im_sym, 200, 255, cv2.THRESH_BINARY)
# Maybe reduce symbol to bounding box?
_, contours, _ = cv2.findContours(cv2.bitwise_not(im_sym_t),2,1)
cnt = contours[1]
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im_sym_t,(x,y),(x+w,y+h),(0,255,0),2)
box = cv2.boxPoints(rect)
box = np.int0(box)

cornerNW = box[1]
cornerNE = box[2]
cornerSW = box[0]
cornerSE = box[3]

print(cornerNW)
print(cornerNE)
print(cornerSW)
print(cornerSE)

cv2.imshow('sym', im_sym_t);

# Search through all the template cards and find the one with the lowest difference
bestError = float("inf")
bestMatch = ""
mypath='../data/cards_png'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
for n in range(0, len(onlyfiles)):
	cur_img = cv2.imread( join(mypath,onlyfiles[n]), cv2.IMREAD_GRAYSCALE )
	# Isolate letter and symbol then threshold
	cur_sym2 = cur_img[18:300, 18:150]
	ret4, cur_sym2_t = cv2.threshold(cur_sym2, 200, 255, cv2.THRESH_BINARY)
	# Find difference between input card and template card
	out = im_sym_t - cur_sym2_t
	error = cv2.countNonZero(out)
	if error < bestError:
		bestError = error
		bestMatch = onlyfiles[n]
print(bestMatch)

# Reminder on how to rotate images, in case I need it
#rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)
#img_rotation = cv2.warpAffine(imgcrop, rotation_matrix, (w, h))


cv2.waitKey(0)
cv2.destroyAllWindows()
