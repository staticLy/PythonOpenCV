'''
Created on 2017��12��5��

@author: liaoyang
'''
import time
import argparse
import datetime
import imutils
import cv2
 
# ������������������������
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# ���video����ΪNone����ô���Ǵ�����ͷ��ȡ����
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
 
# �������Ƕ�ȡһ����Ƶ�ļ�
else:
    camera = cv2.VideoCapture(args["video"])
 
# ��ʼ����Ƶ���ĵ�һ֡
firstFrame = None

# ������Ƶ��ÿһ֡
while True:
    # ��ȡ��ǰ֡����ʼ��occupied/unoccupied�ı�
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
 
    # �������ץȡ��һ֡��˵�����ǵ�����Ƶ�Ľ�β
    if not grabbed:
        break
 
    # ������֡�Ĵ�С��ת��Ϊ�ҽ�ͼ���Ҷ�����и�˹ģ��
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    # �����һ֡��None��������г�ʼ��
    if firstFrame is None:
        firstFrame = gray
        continue

    # ���㵱ǰ֡�͵�һ֡�Ĳ�ͬ
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
    # ��չ��ֵͼ�����׶���Ȼ���ҵ���ֵͼ���ϵ�����
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
 
    # ��������
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        # ���������ı߽���ڵ�ǰ֡�л����ÿ�
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    # draw the text and timestamp on the frame
    # �ڵ�ǰ֡��д�����Լ�ʱ���
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
    #��ʾ��ǰ֡����¼�û��Ƿ��°���
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) &amp; 0xFF
 
    # ���q�������£�����ѭ��
    if key == ord("q"):
        break
 
# �����������Դ���رմ򿪵Ĵ���
camera.release()
cv2.destroyAllWindows()
