import os
import cv2 as cv
import numpy as np

#face recognition

people = ['Deepeka','Priyanka Ch','Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna']
DIR = r'C:\Users\HP\Desktop\CODE\PYTHON .. OPENCV\FAC\Faces\train'

face_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training Done----------------')

# checking length 

# print(f'length of the features list={len(features)}')
# print(f'length of the labels list={len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the feature list and the labels list

face_recognizer.train(features,labels)

face_recognizer.save('faceTrained.yml') 
# saving the trained model 
np.save('features.npy',features)
np.save('labels.npy',labels)
