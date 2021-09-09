import os
import pandas as pd
from pandas.core.base import DataError
import cv2
import shutil


csvframe=pd.read_table("./2007_test.txt",header = None)
label_path = './VOCdevkit/VOC2007/labels/'
list = os.listdir(label_path)
num = len(csvframe)
for i in range(num):
    data = csvframe.iloc[i]
    name = data[0]
    base = os.path.basename(name)
    xml_name = base.split('.jpg')[0]+'.txt'
    img = cv2.imread('./VOCdevkit/VOC2007/JPEGImages/'+base)
    cv2.imwrite('./Garbage/images/test2020/'+base,img)
    shutil.copy(label_path+xml_name,'./Garbage/labels/test2020/'+xml_name)
print("successful!")