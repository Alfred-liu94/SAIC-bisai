import numpy as np
import os
import cv2
import thr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cap = cv2.VideoCapture(1)
M,Minv = thr.get_M_Minv()
#while (cap.isOpened()):
    retLeftUp, frameLeftUp = cap.read()
    unpted = cv2.cvtColor(frameLeftUp, cv2.COLOR_BGR2RGB)
    pted = cv2.warpPerspective(unpted, M, unpted.shape[1::-1], flags=cv2.INTER_LINEAR)
    frameUp = np.hstack((frameLeftUp, pted))
    #ret, frame = cap.read()

    cv2.imshow('frame', frameUp)
    #if cv2.waitKey(40) & 0xFF == ord('q'):
        #break

cap.release()
cv2.destroyAllWindows()
