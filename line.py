import glob
import cv2


img = cv2.imread(fname) #source image
    #我用的图片太大，缩小了一半
#img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转灰度
    #cv2.imshow('img',gray)
    #cv2.waitKey(1000)
    #寻找角点，存入corners，ret是找到角点的flag
    ret, corners = cv2.findChessboardCorners(gray,(5,5),None)