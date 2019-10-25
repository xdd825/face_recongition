#-*- coding: utf-8 -*-

import cv2
import sys
import gc
from face_train import Model

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
        
    #載入模型
    model = Model()
    model.load_model(file_path = './model/liziqiang.face.model.h5')    
              
    #框住人臉的矩形邊框顏色       
    color = (0, 255, 0)
    
    #捕獲指定攝像頭的實時視訊流
    cap = cv2.VideoCapture(0)
    
    #人臉識別分類器本地儲存路徑
    cascade_path = "H:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"    
    
    #迴圈檢測識別人臉
    while True:
        ret, frame = cap.read()   #讀取一幀視訊
        
        if ret is True:
            
            #影象灰化，降低計算複雜度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        #使用人臉識別分類器，讀入分類器
        cascade = cv2.CascadeClassifier(cascade_path)                

        #利用分類器識別出哪個區域為人臉
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #擷取臉部影象提交給模型識別這是誰
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   
                
                #如果是“我”
                if faceID == 0:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                    #文字提示是誰
                    cv2.putText(frame,'liziqiang', 
                                (x + 30, y + 30),                      #座標
                                cv2.FONT_HERSHEY_SIMPLEX,              #字型
                                1,                                     #字號
                                (255,0,255),                           #顏色
                                2)                                     #字的線寬
                else:
                    pass
                            
        cv2.imshow("識別朕", frame)
        
        #等待10毫秒看是否有按鍵輸入
        k = cv2.waitKey(10)
        #如果輸入q則退出迴圈
        if k & 0xFF == ord('q'):
            break

    #釋放攝像頭並銷燬所有視窗
    cap.release()
    cv2.destroyAllWindows()