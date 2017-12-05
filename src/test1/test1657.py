'''
Created on 2017年12月4日

@author: liaoyang
'''
# -*- coding: UTF-8 -*-

import sys

import cv2

#    __Desc__ = 人脸检测小例子，以圆圈圈出人脸


# 待检测的图片路径

imagepath = r'./timg3.jpg'

 

# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
#face_cascade =cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(r'./haarcascade_eye.xml')

 

# 读取图片

image = cv2.imread(imagepath)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

 

# 探测图片中的人脸

faces = face_cascade.detectMultiScale(

    gray,

    scaleFactor = 1.15,

    minNeighbors = 10,

    minSize = (15,15),

    flags = cv2.cv.CV_HAAR_SCALE_IMAGE

)


print("发现{0}个人脸!".format(len(faces)))

for (x,y,w,h) in faces:
    print(x,y,w,h)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (x_eye,y_eye,w_eye,h_eye) in eyes:
        print('eye:',x_eye,y_eye,w_eye,h_eye)
        center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
        radius = int(0.3 * (w_eye + h_eye))
        color = (0, 255, 0)
        thickness = 2
        cv2.circle(roi_color, center, radius, color, thickness)

#for(x,y,w,h) in faces:

    # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

    #cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)

res=cv2.resize(image,(1000,600),interpolation=cv2.INTER_CUBIC)
cv2.imshow('iker',res)
#cv2.imshow('image',image)
cv2.waitKey(0)

#cv2.imshow("Find Faces!",image)

#cv2.waitKey(0)
