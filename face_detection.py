import cv2
import numpy as np

face_xml = cv2.CascadeClassifier('face.xml')
eye_xml = cv2.CascadeClassifier('eye.xml')

cap = cv2.VideoCapture(0)

while True:
    _,frame=cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_xml.detectMultiScale(gray)
    for (x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        r_gray=gray[x:x+w,y:y+h]
        r_color= frame[x:x+w,y:y+h]
        eyes=eye_xml.detectMultiScale(r_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(r_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    cv2.imshow('faceDisplay',frame)
    k = cv2.waitKey(27) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
