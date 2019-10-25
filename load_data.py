import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64

#按照指定影象大小調整尺寸
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #獲取影象尺寸
    h, w, _ = image.shape
    
    #對於長寬不相等的圖片，找到最長的一邊
    longest_edge = max(h, w)    
    
    #計算短邊需要增加多上畫素寬度使其與長邊等長
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    #RGB顏色
    BLACK = [0, 0, 0]
    
    #給影象增加邊界，是圖片長、寬等長，cv2.BORDER_CONSTANT指定邊界顏色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #調整影象大小並返回
    return cv2.resize(constant, (height, width))
#讀取訓練資料
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        #從初始路徑開始疊加，合併成可識別的操作路徑
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path):    #如果是資料夾，繼續遞迴呼叫
            read_path(full_path)
        else:   #檔案
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)                
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                
                #放開這個程式碼，可以看到resize_image()函式的實際呼叫效果
                #cv2.imwrite('1.jpg', image)
                
                images.append(image)                
                labels.append(path_name)                                
                    
    return images,labels
#從指定路徑讀取訓練資料
def load_dataset(path_name):
    images,labels = read_path(path_name)    
    
    #將輸入的所有圖片轉成四維陣列，尺寸為(圖片數量*IMAGE_SIZE*IMAGE_SIZE*3)
    #我和閨女兩個人共1200張圖片，IMAGE_SIZE為64，故對我來說尺寸為1200 * 64 * 64 * 3
    #圖片為64 * 64畫素,一個畫素3個顏色值(RGB)
    images = np.array(images)
    print(images.shape)    
    
    #標註資料，'liziqiang'資料夾下都是我的臉部影象，全部指定為0，另外一個資料夾下是同學的，全部指定為1
    labels = np.array([0 if label.endswith('liziqiang') else 1 for label in labels])    
    
    return images, labels

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        images, labels = load_dataset("/Users/ben/Desktop/data")