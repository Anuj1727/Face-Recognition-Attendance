import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date

path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            today = date.today()
            f.writelines(f'\n{name},{today},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,facloc in zip(encodesCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = facloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4  #upr image ko resize kiya hai isiliye according to the article yaha pr *4 krne ke bad si location of the box ayegi !
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


# faceloc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
#
# facelocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)
#
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)