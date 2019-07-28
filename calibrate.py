import thr
import cv2


cal_imgs = thr.get_images_by_dir('qipange')
object_points, img_points = thr.get_obj_img_points(cal_imgs, grid = (4,4))
test_imgs = cv2.imread("videos/90.1.jpg")
dist = thr.cal_undistort(test_imgs, object_points, img_points)
cv2.imshow('undist', dist)
cv2.waitKey(0)

img = dist

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyAllWindows()
        break
