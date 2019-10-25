#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import sys

from PIL import Image

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    
    #視訊來源，可以來自一段已存好的視訊，也可以直接來自USB攝像頭
    cap = cv2.VideoCapture(camera_idx)                
    
    #告訴OpenCV使用人臉識別分類器
    classfier = cv2.CascadeClassifier("/Volumes/D/Anaconda/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
    classfier.load('/Volumes/D/Anaconda/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
    #識別出人臉後要畫的邊框的顏色，RGB格式
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        ok, frame = cap.read() #讀取一幀資料
        if not ok:            
            break                
    
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #將當前楨影象轉換成灰度影象            
        
        #人臉檢測，1.2和2分別為圖片縮放比例和需要檢測的有效點數
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        
        if len(faceRects) > 0:          #大於0則檢測到人臉                                   
            for faceRect in faceRects:  #單獨框出每一張人臉
                x, y, w, h = faceRect                        
                
                #將當前幀儲存為圖片
                img_name = '%s/%d.jpg'%(path_name, num)                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):   #如果超過指定最大儲存數量退出迴圈
                    break
                
                #畫出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                #顯示當前捕捉到了多少人臉圖片了，這樣站在那裡被拍攝時心裡有個數，不用兩眼一抹黑傻等著
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        #超過指定最大儲存數量結束程式
        if num > (catch_pic_num): break                
                       
        #顯示影象
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    #釋放攝像頭並銷燬所有視窗
    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("擷取人臉", 0, 1000, '/Users/ben/Desktop/data/data_1')


