
import cv2

cap = cv2.VideoCapture(1)
i = 0
while (1):
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('E:/A4' + str(i) + '.jpg', frame)
        i += 1
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()
