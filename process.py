import numpy as np
import cv2
import socket
import thr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

M,Minv = thr.get_M_Minv()
udp = socket.socket(socket. AF_INET, socket.SOCK_DGRAM)
udp.bind(("", 7788))
cal_imgs = thr.get_images_by_dir('qipange')
object_points, img_points = thr.get_obj_img_points(cal_imgs, grid = (4,4))

def thresholding(img):
  x_thresh = thr.abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=230)
  mag_thresh = thr.mag_thresh(img, sobel_kernel=9, mag_thresh=(80, 130))
  dir_thresh = thr.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
  hls_thresh = thr.hls_select(img, thresh=(100, 200))
  lab_thresh = thr.lab_select(img, thresh=(180, 200))
  luv_thresh = thr.luv_select(img, thresh=(200, 240))
  # Thresholding combination
  thresholded = np.zeros_like(x_thresh)
  thresholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (
            luv_thresh == 1)] = 1

  return thresholded


cap = cv2.VideoCapture('videos/Rolling.mp4')
#test_imgs = cv2.imread("videos/14.jpg")


while (cap.isOpened()):
    ret, frame = cap.read()
    dist = thr.cal_undistort(frame, object_points, img_points)
    test_imgs = cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)
    wraped = cv2.warpPerspective(test_imgs, M, test_imgs.shape[1::-1], flags=cv2.INTER_LINEAR)
    thresholded_wraped = thresholding(wraped)
    left_fit, right_fit, left_lane_inds, right_lane_inds = thr.find_line(thresholded_wraped)
    curvature, distance_from_center = thr.calculate_curv_and_pos(thresholded_wraped, left_fit, right_fit)
    result = thr.draw_area(test_imgs, thresholded_wraped, Minv, left_fit, right_fit)
    result_data = thr.draw_values(result, curvature, distance_from_center)
    #print(wraped.shape[0])
    cv2.imshow("result", result_data)
    d_q, d_b, d_s, d_g = thr.data_out(int(abs(distance_from_center)), curvature)
    #print(distance_from_center)
    d_out = bytes([d_q, 0, d_b, d_s, d_g])
    udp.sendto(d_out, ('192.168.43.180', 25000))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#socket.close()
cap.release()
cv2.destroyAllWindows()

#plt.imshow(thresholded_wraped)
#plt.show()