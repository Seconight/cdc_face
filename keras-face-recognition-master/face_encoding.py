import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1
import sys


mtcnn_model = mtcnn()  #创建mtcnn对象检测图片中的人脸
threshold = [0.5,0.8,0.9]  #门限

#载入facenet将检测到的人脸转化为128维的向量
facenet_model = InceptionResNetV1()
model_path = './model_data/facenet_keras.h5'
facenet_model.load_weights(model_path)
# faceImage
img = cv2.imread('./userFace/'+sys.argv[1]+'.jpg')    #读取对应的图像
# print(facePath)
# img = cv2.imread(facePath)    #读取对应的图像
#cv2.imshow(img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

rectangles = mtcnn_model.detectFace(img, threshold)  # 利用facenet_model检测人脸

# 转化成正方形
rectangles = utils.rect2square(np.array(rectangles))
# facenet要传入一个160x160的图片
rectangle = rectangles[0]
# 人脸的五个关键点
landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
# 截下人脸图
crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
crop_img = cv2.resize(crop_img,(160,160))
# 进行人脸对齐
new_img,_ = utils.Alignment_1(crop_img,landmark)
# 增加维度
new_img = np.expand_dims(new_img,0)
# 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
face_encoding = utils.calc_128_vec(facenet_model,new_img)
enc2str = ",".join(str(li) for li in face_encoding.tolist())
# 设置输出路径
encodingPath = './userFace/encoding'+sys.argv[1]+'.txt'
f=open(encodingPath,'w')
f.write(enc2str)
f.close()

print("finished")