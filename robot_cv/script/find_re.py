
import cv2

cap=cv2.VideoCapture(0)
while (cap.isOpened()):

    ret,frame=cap.read()
    if not ret:  
        print("无法接收帧（流可能已结束）")  
        break
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(5) >= 0:
        break
   
    #cv2.imshow('frame1',img)
              #
       
cap.release()  
cv2.destroyAllWindows()
