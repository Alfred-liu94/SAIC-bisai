import numpy as np
import os
import cv2
import thr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

M,Minv = thr.get_M_Minv()


def thresholding(img):
  x_thresh = thr.abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=230)
  mag_thresh = thr.mag_thresh(img, sobel_kernel=9, mag_thresh=(80, 1330))
  dir_thresh = thr.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
  hls_thresh = thr.hls_select(img, thresh=(100, 200))
  lab_thresh = thr.lab_select(img, thresh=(180, 200))
  luv_thresh = thr.luv_select(img, thresh=(200, 240))
  # Thresholding combination
  thresholded = np.zeros_like(x_thresh)
  thresholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (
            luv_thresh == 1)] = 1

  return thresholded


test_imgs = cv2.imread("videos/50.jpg")
wraped = cv2.warpPerspective(test_imgs, M, test_imgs.shape[1::-1], flags=cv2.INTER_LINEAR)
thretest = thresholding(wraped)
plt.imshow(thretest)
plt.show()