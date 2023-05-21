import numpy as np
import cv2 as cv

face_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")

people = ['Deepeka','Priyanka_Ch','ben_afflek', 'elton_john', 'jerry_seinfeld','madonna']

features = np.load('features.npy',allow_pickle=True)
labels = np.load('labels.npy',allow_pickle=True)

#deep computer vision
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faceTrained.yml')

img = cv.imread('C:/Users/HP/Desktop/CODE/PYTHON .. OPENCV/FAC/Faces/val/Deepeka/3.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

#Detect the face in the image
faces_rect = face_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected face',img)

cv.waitKey(0)