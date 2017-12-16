import cv2
import numpy as np
import matplotlib.pyplot as plt 
from os import listdir
from os.path import isfile, join
import math

img = cv2.imread('../data/input/2c.jpg', cv2.IMREAD_GRAYSCALE)

# First, we'll threshold the image and make it a binary image 
_,imthresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# Perform opening on thresholded image to remove noise in space
kernel = np.ones((6,6), np.uint8)
imthresh_open = cv2.morphologyEx(imthresh, cv2.MORPH_OPEN, kernel)

# Get 4 corner points using rotated rectangle
_, contours, _ = cv2.findContours(imthresh_open,2,1)
cnt = contours[0]

#Find largest contour
cnt_max = 0
for contour in contours:
	if cv2.contourArea(contour) > cnt_max:
		cnt_max = cv2.contourArea(contour)
		cnt = contour
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

cornerNW = box[1]
cornerNE = box[2]
cornerSW = box[0]
cornerSE = box[3]

# Correct corner points that might orient the image sideways
dist_NW_to_NE = math.hypot(cornerNE[1] - cornerNW[1], cornerNE[0] - cornerNW[0])
dist_NE_to_SE = math.hypot(cornerSE[1] - cornerNE[1], cornerSE[0] - cornerNE[0])
if dist_NW_to_NE > dist_NE_to_SE:
	cornerNW = box[2]
	cornerNE = box[3]
	cornerSW = box[1]
	cornerSE = box[0]

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

# Do some homography stuff to "straighten" the card
pts_src = np.array([cornerNW, cornerNE, cornerSW, cornerSE])
pts_dst = np.array([[0.0, 0.0], [691, 0.0], [0.0, 1056],[691, 1056]])
im_dst = np.zeros((1056, 691, 1), np.uint8)
hom, status = cv2.findHomography(pts_src, pts_dst)
im_hom = cv2.warpPerspective(imthresh, hom, (im_dst.shape[1],im_dst.shape[0]))
cv2.imshow('im_hom', im_hom);

# Isolate the card letter and symbol
im_suite_rank = im_hom[18:600, 18:99]
_, im_suite_rank_thresh = cv2.threshold(im_suite_rank, 200, 255, cv2.THRESH_BINARY)
im_suite_rank_thresh_open = cv2.morphologyEx(im_suite_rank_thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow('im_suite_rank_thresh_open', im_suite_rank_thresh_open)

# Find letter and symbol bounding boxes
_, contours, _ = cv2.findContours(cv2.bitwise_not(im_suite_rank_thresh_open),cv2.RETR_EXTERNAL,1)
cnt_suite = contours[0]
cnt_rank = contours[1]
suite_x, suite_y, suite_w, suite_h = cv2.boundingRect(cnt_suite)
rank_x, rank_y, rank_w, rank_h = cv2.boundingRect(cnt_rank)

# Crop letter and symbol into two images, resize both to a set standard
im_suite = im_suite_rank_thresh[suite_y:(suite_y + suite_h), suite_x:(suite_x + suite_w)]
im_rank = im_suite_rank_thresh[rank_y:(rank_y + rank_h), rank_x:(rank_x + rank_w)]
im_suite_resized = cv2.resize(im_suite, (100,100))
im_rank_resized = cv2.resize(im_rank, (100,100))
_, im_suite_thresh = cv2.threshold(im_suite_resized, 200, 255, cv2.THRESH_BINARY)
_, im_rank_thresh = cv2.threshold(im_rank_resized, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('im_rank_thresh', im_rank_thresh)
cv2.imshow('im_suite_thresh', im_suite_thresh)

# Search through all the template cards and find the one with the lowest difference
bestError = float("inf")
bestMatch = ""
mypath='../data/cards_png'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
for n in range(0, len(onlyfiles)):
	cur_img = cv2.imread( join(mypath,onlyfiles[n]), cv2.IMREAD_GRAYSCALE )
	
	# Isolate letter and symbol then threshold
	cur_suite_rank = cur_img[18:300, 18:105]
	_, cur_suite_rank_thresh = cv2.threshold(cur_suite_rank, 200, 255, cv2.THRESH_BINARY)
	cur_suite_rank_thresh_open = cv2.morphologyEx(cur_suite_rank_thresh, cv2.MORPH_OPEN, kernel)
	
	# Find letter and symbol bounding boxes
	_, cur_contours, _ = cv2.findContours(cv2.bitwise_not(cur_suite_rank_thresh_open), cv2.RETR_EXTERNAL, 1)
	cur_cnt_suite = cur_contours[0]
	cur_cnt_rank = cur_contours[1] 
	cur_suite_x, cur_suite_y, cur_suite_w, cur_suite_h = cv2.boundingRect(cur_cnt_suite)
	cur_rank_x, cur_rank_y, cur_rank_w, cur_rank_h = cv2.boundingRect(cur_cnt_rank)

	# Crop letter and symbol into two images, resize both to a set standard
	cur_suite = cur_suite_rank_thresh[cur_suite_y:(cur_suite_y + cur_suite_h), cur_suite_x:(cur_suite_x + cur_suite_w)]
	cur_rank = cur_suite_rank_thresh[cur_rank_y:(cur_rank_y + cur_rank_h), cur_rank_x:(cur_rank_x + cur_rank_w)]
	cur_suite_resized = cv2.resize(cur_suite, (100,100))
	cur_rank_resized = cv2.resize(cur_rank, (100,100))
	_, cur_suite_thresh = cv2.threshold(cur_suite_resized, 200, 255, cv2.THRESH_BINARY)
	_, cur_rank_thresh = cv2.threshold(cur_rank_resized, 200, 255, cv2.THRESH_BINARY)

	# Find difference between input card and template card
	diff_suite = im_suite_thresh - cur_suite_thresh
	diff_rank = im_rank_thresh - cur_rank_thresh

	# if onlyfiles[n] == "KS.png":
	# 	cv2.imshow("cur_suite_rank_thresh_open", cur_suite_rank_thresh_open)
	# 	cv2.imshow("diff_suite", diff_suite)
	# 	cv2.imshow("cur_suite_thresh", cur_suite_thresh)
	# 	cv2.imshow("cur_rank_thresh", cur_rank_thresh)

	error = cv2.countNonZero(diff_suite) + cv2.countNonZero(diff_rank)
	if error < bestError:
		bestError = error
		bestMatch = onlyfiles[n]

print(bestMatch)
cv2.waitKey(0)
cv2.destroyAllWindows()
