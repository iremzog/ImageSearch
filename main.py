#!/usr/bin/python

import numpy as np
import cv2
import sys

'''
Takes two argument, first one is image file that contains the star map, and the second one is the patch from this star map.
'''
def find_star_location(starmap, smallstar):

  good_match_ratio = 0.3

  sift = cv2.xfeatures2d.SIFT_create()
  keypoints_1, desc_1= sift.detectAndCompute(starmap,None)
  keypoints_2, desc_2 = sift.detectAndCompute(smallstar,None)

  bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
  matches = bf.match(desc_1,desc_2)
  matches = sorted(matches, key = lambda x:x.distance)

  numGoodMatches = int(len(matches) * good_match_ratio)
  matches = matches[:numGoodMatches]

  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints_1[match.queryIdx].pt
    points2[i, :] = keypoints_2[match.trainIdx].pt

  h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

  x, y, _ = smallstar.shape
  old_points = np.array([
                        [0,0,1],
                        [y,0,1],
                        [0,x,1],
                        [y,x,1]
  ])
  new_points = np.dot(old_points, h.T)
  new_points = new_points.astype('int64')

  return new_points[:,:-1]


if __name__ == '__main__':
  image_path_1 = sys.argv[1]
  image_path_2 = sys.argv[2]

  starmap = cv2.imread(image_path_1)
  smallstar = cv2.imread(image_path_2)

  print(find_star_location(starmap, smallstar))
