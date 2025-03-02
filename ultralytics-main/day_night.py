import cv2
import numpy as np
import os
import shutil
# 计算明暗程度平均值函数
def col_v(img):
    img = cv2.imread(img) # 读取传进来的图片
        
    if img is None:
        return 0

    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # 转换图片为HSV颜色模式
    h,s,v = cv2.split(img) # 分割后会得到HSV3个通道的值
    v = np.average(v) # 对V（明暗度）求平均值，也可以用np.sum/图片像素点总数来求平均值
    return v
 

root='/home/data2/mxl/ultralytics/ultralytics-main/dataset/onlycar_board2/images/val' 
rootx='/home/data2/mxl/ultralytics/ultralytics-main/dataset/onlycar_board2/labels/val' 

root2='/home/data2/mxl/ultralytics/ultralytics-main/dataset/night/images'
root2x='/home/data2/mxl/ultralytics/ultralytics-main/dataset/night/labels'

for i in os.listdir(root):
    v_2 = col_v(os.path.join(root,i))
    if v_2 ==0:
        continue
    print(v_2)
    # 我们假设根据经验，平均值大于100的是白天
    if v_2>50:
        print('day')
    else:
        print('night')
        shutil.copy(os.path.join(root,i),os.path.join(root2,i))
        shutil.copy(os.path.join(rootx,i[:-4]+'.txt'),os.path.join(root2x,i[:-4]+'.txt'))
        