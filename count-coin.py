import cv2

import numpy as np


def count_coin(fileName):
    # 读取,灰色图片
    img = cv2.imread(fileName)
    
    # 将图像,转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 对图像进行高斯模糊处理，以减少噪声
    img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
    
    cv2.imshow('111',img_blur)
    cv2.waitKey(0)

    # 获得,卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 腐蚀
    # img_blur = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, kernel)
    img_blur = cv2.erode(img_blur, kernel, iterations = 1)
    cv2.imshow('111',img_blur)
    cv2.waitKey(0)

    # 使用Canny边缘,检测寻找图像中的边缘
    edges = cv2.Canny(img_blur, 30, 150)
    cv2.imshow('111',edges)
    cv2.waitKey(0)

    # 进行,圆检测
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=80)

    # print(circles.shape)
    num = 0
    
    # 如果检测到圆，则计算硬币数量
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # 在图像上,绘制检测到的圆和半径
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
            
            num += 1
    
    # 显示结果图像和硬币数量
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    print("num = ", num)


fileName = "/home/hdy/图片/coin/6.png"
fileName = "/home/hdy/图片/coin/30.jpg"
fileName = "/home/hdy/图片/coin/4.jpg"

count_coin(fileName)

